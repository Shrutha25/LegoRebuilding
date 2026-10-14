import pandas as pd


def df(rows, cols):
    return pd.DataFrame(rows, columns=cols, dtype=str)


def fixture():
    rels = df([["3001pr1", "3001", "P"], ["3001pr2", "3001pr1", "P"], ["9999", "1111", "R"]],
              ["child_part_num", "parent_part_num", "rel_type"])
    themes = df([["1", "Town", ""], ["2", "Supplemental", ""], ["3", "Bulk Sub", "2"]],
                ["id", "name", "parent_id"])
    sets = df([["10-1", "Small House", "2020", "1", "http://img/10"],
               ["20-1", "Parts Pack", "2021", "2", "http://img/20"],
               ["30-1", "Sub Pack", "2021", "3", "http://img/30"]],
              ["set_num", "name", "year", "theme_id", "img_url"])
    colors = df([["1", "Red"], ["2", "Blue"]], ["id", "name"])
    inventories = df([["100", "1", "10-1"], ["101", "2", "10-1"], ["200", "1", "20-1"],
                      ["300", "1", "30-1"]], ["id", "version", "set_num"])
    inv_parts = df([
        ["100", "3001", "1", "99", "f", ""],                          # old version: ignored
        ["101", "3001pr1", "1", "2", "f", "http://img/3001-red"],     # variant -> 3001
        ["101", "3002", "2", "3", "f", "http://img/3002-blue"],
        ["101", "3003", "2", "5", "t", ""],                           # spare: excluded
        ["200", "3001", "1", "10", "f", ""],
        ["300", "3001", "1", "10", "f", ""],
    ], ["inventory_id", "part_num", "color_id", "quantity", "is_spare", "img_url"])
    parts = df([["3001", "Brick 2 x 4"], ["3001a", "Brick 2 x 4 old"], ["3002", "Brick 2 x 3"],
                ["3003", "Brick 2 x 2"], ["41879a", "Legs Short"], ["3001pr1", "Brick printed"]],
               ["part_num", "name"])
    return {"part_relationships": rels, "themes": themes, "sets": sets, "colors": colors,
            "inventories": inventories, "inventory_parts": inv_parts, "parts": parts}


def write_rb_dir(path):
    """Write the fixture as Rebrickable-style CSVs; returns the directory."""
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in fixture().items():
        frame.to_csv(path / f"{name}.csv", index=False)
    return path
