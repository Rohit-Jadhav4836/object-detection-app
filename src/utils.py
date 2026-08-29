import pandas as pd

def detections_to_dataframe(detections: list[dict]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["class", "confidence", "bbox"])
    df = pd.DataFrame(detections)
    df["confidence"] = (df["confidence"] * 100).round(1)
    return df.sort_values("confidence", ascending=False)


def summarize_detections(detections: list[dict]) -> dict:
    if not detections:
        return {"total": 0, "unique_classes": 0, "top_class": "None"}
    classes = [d["class"] for d in detections]
    class_counts = pd.Series(classes).value_counts()
    return {
        "total": len(detections),
        "unique_classes": len(class_counts),
        "top_class": class_counts.index[0],
    }


def filter_detections_by_class(detections: list[dict], selected_classes: list[str]) -> list[dict]:
    """Filter detections to only include selected classes."""
    if not selected_classes:
        return detections
    return [d for d in detections if d["class"] in selected_classes]