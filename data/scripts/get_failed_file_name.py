import yaml
import os

def get_failed_motion_files(yaml_path, txt_path):
    # Translated comment.
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Translated comment.
    idx_to_file = {}
    for motion in data['motions']:
        idx = motion['idx']
        # Translated comment.
        filename = os.path.splitext(os.path.basename(motion['file']))[0]
        idx_to_file[idx] = filename
    
    # Translated comment.
    with open(txt_path, 'r') as f:
        failed_indices = [int(line.strip()) for line in f if line.strip()]
    
    # Translated comment.
    failed_files = []
    for idx in failed_indices:
        if idx in idx_to_file:
            failed_files.append(idx_to_file[idx])
        else:
            print(f"Warning: Index {idx} not found in YAML file")
    
    return failed_files

# Translated comment.
if __name__ == "__main__":
    yaml_file = "data/yaml_files/video_convert.yaml"
    txt_file = "failed_motions_video_convert.txt"
    
    result = get_failed_motion_files(yaml_file, txt_file)
    
    print("Failed motion files:")
    with open("video_convert_faile_file_name.txt", 'w') as f:
        for filename in result:
            print(filename)
            f.write(filename + "\n")