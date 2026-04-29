import re
from ruamel.yaml import YAML

# Translated comment.
# failed_indices = set()
# with open('results/high_dynamic_terrain/failed_motions.txt', 'r') as f:
#     for line in f:
#         line = line.strip()
#         match = re.match(r'^(\d+)=', line)
#         if match:
#             failed_indices.add(str(match.group(1)))
failed_list = []
with open("data/aist++/ignore_list.txt",  'r') as f:
    for line in f:
        line = line.strip()
        failed_list.append(line)
failed_indices = set(failed_list)
print(f"Failed indices: {failed_indices}")
print("len(failed_indices):", len(failed_indices))

# Translated comment.
yaml = YAML()
with open('data/yaml_files/aist++.yaml', 'r') as f:
    data = yaml.load(f)

# Translated comment.
filtered_motions = [m for m in data['motions'] if m['file'].split("/")[-1].replace(".npy", "") not in failed_indices]
for new_idx, motion in enumerate(filtered_motions):
    motion['idx'] = new_idx
data['motions'] = filtered_motions
print(f"Filtered motions: {len(data['motions'])}")

# Translated comment.
with open('data/yaml_files/aist++_clean.yaml', 'w') as f:
    yaml.dump(data, f)