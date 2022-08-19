
# FINAL GC
final_gc = [600, 500, 400, 350, 300, 260, 220, 200, 180, 160, 140, 130, 120, 110, 100, 90, 80,
            70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

assert len(final_gc) == 30

final_points = [120, 100, 80, 60, 40, 30, 20, 15, 10, 5]
final_kom = final_points[:]
assert len(final_points) == 10

stage_results = [220, 180, 160, 140, 120, 110, 95, 80, 70, 60, 50, 40, 35, 30, 25, 20, 16, 12, 8, 4]


assert len(stage_results) == 20
gc_results = [30, 26, 22, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
assert len(gc_results) == 20
point_results = [12, 8, 6, 4, 2, 1]
assert len(point_results) == 6
mountain_results = point_results[:]

intermediate_sprints = [20, 16, 12, 8, 6, 5, 4, 3, 2, 1]
HC_climb = [30, 25, 20, 15, 10, 6, 4, 2]
