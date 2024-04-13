import os
import stat
import concurrent.futures
from tqdm.auto import tqdm
import dotenv

dotenv.load_dotenv(override=False)


def change_permission(path, progress, current_user_id):
    try:
        if os.stat(path).st_uid == current_user_id:
            current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
            os.chmod(path, current_permissions | stat.S_IWGRP)
    except Exception as e:
        print(f"Error changing permissions for {path}: {e}")
    finally:
        progress.update(1)
        progress.refresh()


def walk_directory(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        paths.append(root)
        for name in dirs + files:
            paths.append(os.path.join(root, name))
    return paths


def parallel_walk(root_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        all_paths = [root_dir]
        futures = []
        progress = tqdm(desc="Walking directories", unit="dir")

        for first_level_dir in os.listdir(root_dir):
            first_level_path = os.path.join(root_dir, first_level_dir)
            if os.path.isdir(first_level_path):
                for second_level_dir in os.listdir(first_level_path):
                    second_level_path = os.path.join(first_level_path, second_level_dir)
                    if os.path.isdir(second_level_path):
                        future = executor.submit(walk_directory, second_level_path)
                        futures.append(future)
                    else:
                        all_paths.append(second_level_path)  # Append second-level files
            all_paths.append(first_level_path)  # Append first-level folders and files

        for future in concurrent.futures.as_completed(futures):
            all_paths.extend(future.result())
            progress.update(1)

        return all_paths


def chmod_parallel(paths, current_user_id):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor, tqdm(
        total=len(paths), desc="Changing permissions", unit="file"
    ) as progress:
        futures = [
            executor.submit(change_permission, path, progress, current_user_id)
            for path in paths
        ]
        concurrent.futures.wait(futures)


def main():
    root_dir = os.path.join(os.environ.get("DATA_DIR"), "attacked")
    print(f"Collecting paths in {root_dir}")
    paths = parallel_walk(root_dir)
    print(f"Changing permissions for all folders and files in {root_dir}")
    chmod_parallel(paths, os.getuid())


if __name__ == "__main__":
    main()
