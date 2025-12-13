from shapely.geometry import box
import matplotlib.pyplot as plt
from src.geometry import analyze_buffers

def make_panel(x, y, w, h):
    return box(x, y, x + w, y + h)

def visualize(result):
    fig, ax = plt.subplots()

    b1 = result["geometry"]["buffer_1"]
    b2 = result["geometry"]["buffer_2"]
    mp = result["geometry"]["merged_panels"]

    ax.plot(*b1.exterior.xy, color="blue", label="Buffer 1")
    ax.plot(*b2.exterior.xy, color="green", label="Buffer 2")

    if mp:
        ax.plot(*mp.exterior.xy, color="red", label="Merged Panels")

    ax.scatter(
        result["geometry"]["center"][0],
        result["geometry"]["center"][1],
        color="black",
        label="Center"
    )

    ax.set_aspect("equal")
    ax.legend()
    plt.show()

# -------------------------
# TEST CASES
# -------------------------

center_x, center_y = 500, 500

# CHANGE THIS TO TEST DIFFERENT SCENARIOS
panels = [
    make_panel(520, 520, 50, 30),
    make_panel(540, 530, 50, 30)  # overlapping panel
]

result = analyze_buffers(
    center_x=center_x,
    center_y=center_y,
    panels=panels
)

print("STATUS:", result["status"])
print("QC:", result["qc_status"])
print("AREA (sqft):", round(result["total_area_sqft"], 2))

visualize(result)
