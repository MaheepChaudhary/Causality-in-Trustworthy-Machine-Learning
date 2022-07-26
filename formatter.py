from copy import deepcopy
import os
import re
import shutil
import markdown
from pathlib import Path


def open_fiile(filename):
    with open(filename, 'r') as f:
        text = f.read()

    return text


def format(text):
    title_text = "- \[\w*\W*\D*\d*\]"
    image_text = "images\/\w*\d*.png"
    destination_folder = "imgs"
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    img_rex = re.compile(title_text)
    papers = re.findall(title_text, text)
    blocks = text.split("- [")
    header = blocks[0]
    blocks = blocks[1:]
    assert len(blocks) == len(papers)
    new_blocks = []
    for paper, block in zip(papers, blocks):
        # get name for expected folder
        new_block = deepcopy(block)
        name = paper.replace(
            "- [", "").replace("]", "").strip().replace(" ", "_").replace("?", "")
        save_dir = os.path.join(destination_folder, name)
        Path(save_dir).mkdir(
            parents=True, exist_ok=True)
        print(name)
        # find all images in the text block
        images = re.findall(image_text, block)
        print(images)
        print("-------------------------")
        for each_image in images:
            shutil.copy(each_image, save_dir)
            print(each_image, save_dir + '/' +
                  os.path.basename(each_image))
            new_block = new_block.replace(each_image, save_dir + '/' +
                                          os.path.basename(each_image.replace("?", "")))

        new_blocks.append(new_block)

    new_readme_text = header + "- [" + "- [".join(new_blocks)
    with open("Readme_formatted.md", 'w') as f:
        f.write(new_readme_text)


text = open_fiile("Readme.md")
format(text)
