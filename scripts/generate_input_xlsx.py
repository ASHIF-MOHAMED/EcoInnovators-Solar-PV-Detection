from openpyxl import Workbook

rows = [
    ("sample_id","lat","lon"),
    (1001, 12.9903, 80.2422),
    (1002, 28.6139, 77.2090),
    (1003, 19.0760, 72.8777),
    (1004, 13.0827, 80.2707),
    (1005, 22.5726, 88.3639),
    (1006, 17.3850, 78.4867),
    (1007, 11.0168, 76.9558),
    (1008, 26.9124, 75.7873),
    (1009, 23.0225, 72.5714),
    (1010, 12.9716, 77.5946),
]

wb = Workbook()
ws = wb.active
for r in rows:
    ws.append(r)

out_path = "input_samples.xlsx"
wb.save(out_path)
print(f"Saved {out_path}")
