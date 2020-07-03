import os
import shutil
from pathlib import Path


def main():
    original_dataset_dir = Path("./datasets/cats_and_dogs/train")
    base_dir = Path("./_datasets/cats_and_dogs")
    train_dir = base_dir / "train"
    validation_dir = base_dir / "validation"
    test_dir = base_dir / "test"
    train_cats_dir = train_dir / "cats"
    train_dogs_dir = train_dir / "dogs"
    validation_cats_dir = validation_dir / "cats"
    validation_dogs_dir = validation_dir / "dogs"
    test_cats_dir = test_dir / "cats"
    test_dogs_dir = test_dir / "dogs"

    base_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_cats_dir.mkdir(parents=True, exist_ok=True)
    train_dogs_dir.mkdir(parents=True, exist_ok=True)
    validation_cats_dir.mkdir(parents=True, exist_ok=True)
    validation_dogs_dir.mkdir(parents=True, exist_ok=True)
    test_cats_dir.mkdir(parents=True, exist_ok=True)
    test_dogs_dir.mkdir(parents=True, exist_ok=True)

    imgs = [f"cat.{i}.jpg" for i in range(2000)]

    for img in imgs[:1000]:
        src = original_dataset_dir / img
        dst = train_cats_dir / img
        shutil.copyfile(str(src), str(dst))

    for img in imgs[1000:1500]:
        src = original_dataset_dir / img
        dst = validation_cats_dir / img
        shutil.copyfile(str(src), str(dst))

    for img in imgs[1000:1500]:
        src = original_dataset_dir / img
        dst = test_cats_dir / img
        shutil.copyfile(str(src), str(dst))

    imgs = [f"dog.{i}.jpg" for i in range(2000)]

    for img in imgs[:1000]:
        src = original_dataset_dir / img
        dst = train_dogs_dir / img
        shutil.copyfile(str(src), str(dst))

    for img in imgs[1000:1500]:
        src = original_dataset_dir / img
        dst = validation_dogs_dir / img
        shutil.copyfile(str(src), str(dst))

    for img in imgs[1000:1500]:
        src = original_dataset_dir / img
        dst = test_dogs_dir / img
        shutil.copyfile(str(src), str(dst))


if __name__ == "__main__":
    main()
