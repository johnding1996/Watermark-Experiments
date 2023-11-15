import pandas as pd


def style_progress_dataframe(row_list):
    dataframe = pd.DataFrame(
        row_list,
        columns=[
            "Dataset",
            "Source",
            "Attack",
            "Strength",
            "Generated",
            "Reversed",
            "Decoded",
            "Measured",
        ],
    ).sort_values(
        by=["Dataset", "Source", "Attack", "Strength"],
        ascending=[True, True, True, True],
    )
    dataframe = dataframe.astype(
        {
            "Dataset": "string",
            "Source": "string",
            "Attack": "string",
            "Strength": "string",
            "Generated": "Int64",
            "Reversed": "Int64",
            "Decoded": "Int64",
            "Measured": "Int64",
        }
    )

    def style_rows_by_status(row):
        if pd.isna(row["Generated"]) or row["Generated"] < 5000:
            return (
                [""] * row.index.get_loc("Generated")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Generated") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_reverse(row):
        if row["Source"].endswith("tree_ring") and (
            pd.isna(row["Reversed"]) or row["Reversed"] < 5000
        ):
            return (
                [""] * row.index.get_loc("Reversed")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Reversed") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_decode(row):
        if pd.isna(row["Decoded"]) or row["Decoded"] < 5000:
            return (
                [""] * row.index.get_loc("Decoded")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Decoded") - 1)
            )
        else:
            return [""] * len(row)

    def style_rows_by_metric(row):
        if not pd.isna(row["Attack"]) and (
            pd.isna(row["Measured"]) or row["Measured"] < 5000
        ):
            return (
                [""] * row.index.get_loc("Measured")
                + ["background-color: lightcoral"]
                + [""] * (len(row) - row.index.get_loc("Measured") - 1)
            )
        else:
            return [""] * len(row)

    styler = (
        dataframe.style.apply(style_rows_by_status, axis=1)
        .apply(style_rows_by_reverse, axis=1)
        .apply(style_rows_by_decode, axis=1)
        .apply(style_rows_by_metric, axis=1)
    )

    return styler
