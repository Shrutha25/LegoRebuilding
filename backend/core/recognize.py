"""Brickognize client (legacy endpoint). Kept in one module so it is easy to swap."""

import cv2
import requests

API_URL = "https://api.brickognize.com/predict/parts/"


def recognize_crop(crop, timeout=15):
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"error": "encode failed"}
    try:
        r = requests.post(
            API_URL,
            params={"predict_color": "true", "top_k_items": 3, "top_k_colors": 1},
            files={"query_image": ("piece.jpg", buf.tobytes(), "image/jpeg")},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def summarize(result):
    """Pull the top part + colour out of a Brickognize response."""
    items = result.get("items") or []
    if not items:
        return None
    top = items[0]
    colors = top.get("colors") or result.get("colors") or []
    return {
        "part_id": top["id"],
        "name": top["name"],
        "score": round(top["score"], 3),
        "color": colors[0]["name"] if colors else None,
        "alternatives": [{"part_id": i["id"], "name": i["name"], "score": round(i["score"], 3)}
                         for i in items[1:]],
    }
