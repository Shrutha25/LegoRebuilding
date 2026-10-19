"""PATCH LAYER over BrickGPT: inventory-constrained sampling. Nothing in repo/ is modified; update BrickGPT by pulling it.

What changes against BrickGPT.generate_brick_with_rejection_sampling / _try_adding_brick / _generate_brick_with_logit_masking:

 1. _try_adding_brick (the accept/reject point): a brick whose type has no stock left is rejected ("no_inventory").
    Remaining stock is always inventory minus the bricks currently in the structure, so a physics rollback (which rebuilds the
    structure from a prefix) restores the counts with no bookkeeping to undo.
 2. _generate_brick_with_logit_masking: same loop as upstream, except the allowed tokens for the dimensions are limited to
    types that still have stock (and EOS stays allowed). Without this the model keeps proposing used-up types and the 500
    rejections per brick make strict runs very slow. (1) stays the authority, (2) is only a speed-up.
 3. generate_brick_with_rejection_sampling: when the rejection budget runs out upstream returns the last, invalid brick;
    here that ends generation instead ("no_valid_brick"), and the reason generation stopped is recorded in end_reason:
      finished       the model emitted EOS itself
      ran_out        no brick of any type is left in the inventory
      no_valid_brick bricks were left but none that fits was found within max_brick_rejections

inventory=None gives plain upstream behaviour (shopping mode / unconstrained).
"""
from collections import Counter

from brickgpt.data import Brick, BrickStructure
from brickgpt.models import BrickGPT, BrickGPTConfig

from inventory import PART_DIMS, dims_part, part_of, type_totals


class InventoryBrickGPT(BrickGPT):
    def __init__(self, cfg: BrickGPTConfig, inventory: dict | None = None):
        super().__init__(cfg)
        self.set_inventory(inventory)

    def set_inventory(self, inventory):
        """inventory: {(part, colour): qty} (colour is ignored here) or None for unconstrained."""
        self.stock = None if inventory is None else type_totals(inventory)
        self.end_reason = None
        self._bricks = None
        self._exhausted = False

    def __call__(self, caption: str) -> dict:
        self.end_reason = None
        out = super().__call__(caption)
        out["end_reason"] = self.end_reason
        return out

    # ---- remaining stock, derived from the structure so rollbacks restore it
    def remaining(self, bricks=None) -> Counter:
        bricks = self._bricks if bricks is None else bricks
        used = Counter(part_of(b) for b in bricks.bricks) if bricks is not None else Counter()
        return Counter({p: n - used[p] for p, n in self.stock.items() if n - used[p] > 0})

    # ---- (1) accept / reject
    def _try_adding_brick(self, brick_str, bricks, rejected_bricks):
        result = super()._try_adding_brick(brick_str, bricks, rejected_bricks)
        if result == "success" and self.stock is not None:
            b = Brick.from_txt(brick_str)
            if self.remaining(bricks)[dims_part(b.h, b.w)] <= 0:
                return "no_inventory"
        return result

    # ---- (3) generation end reason
    def generate_brick_with_rejection_sampling(self, prompt=None, bricks=None):
        if bricks is None:
            bricks = BrickStructure([])
        self._bricks = bricks
        self._exhausted = False
        brick, reasons = super().generate_brick_with_rejection_sampling(prompt, bricks)
        if brick and self.stock is not None and self._try_adding_brick(brick, bricks, set()) != "success":
            self.end_reason = "no_valid_brick"        # upstream gave up and handed back an invalid brick
            reasons["gave_up"] += 1
            return "", reasons
        if not brick:
            self.end_reason = "ran_out" if self._exhausted else "finished"
        return brick, reasons

    # ---- (2) token masking by stock
    def _generate_brick_with_logit_masking(self, prompt=None, temperature=None):
        if self.stock is None:
            return super()._generate_brick_with_logit_masking(prompt, temperature)
        if temperature is None:
            temperature = self.temperature
        left = self.remaining()
        if not left:
            self._exhausted = True
            return ""                                  # nothing left to place: ends like EOS, before any LLM call
        pairs = [PART_DIMS[p] for p in left]          # (short, long) of each type with stock
        sizes = {s for pr in pairs for s in pr}
        allowed_h = tuple(str(s) for s in sorted(sizes)) + (self.llm.tokenizer.eos_token,)
        posns = tuple(str(i) for i in range(self.world_dim))

        result_ids, h = [], None
        for step, allowed in enumerate([allowed_h, ("x",), None, (" (",), posns, (",",), posns, (",",), posns, (")\n",)]):
            if step == 2:                              # width: only those that make a type with stock together with h
                allowed = tuple(str(w) for w in sorted({a if b == h else b for a, b in pairs if h in (a, b)}))
            next_token_id = self.llm(
                prompt, return_as_ids=True, max_new_tokens=1, temperature=temperature, top_k=None, top_p=None,
                logits_processor=self._build_allow_tokens_logits_processor(allowed))[0]
            result_ids.append(next_token_id)
            if next_token_id == self.llm.tokenizer.eos_token_id:
                break
            if step == 0:
                h = int(self.llm.tokenizer.decode([next_token_id]))
            prompt = None
        return self.llm.tokenizer.decode(result_ids, skip_special_tokens=True)
