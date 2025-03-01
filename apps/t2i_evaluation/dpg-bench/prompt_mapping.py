import os
import csv

def prompt_filename_mapping(data_dir):

    data = []
    csv_file = f"./dpg_prompt_map.csv"

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        name = os.path.splitext(filename)[0]

        row = (content, name)
        data.append(row)

    print (len(data))

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for row in data:
            writer.writerow(row)



if __name__ == '__main__':
    prompt_filename_mapping('./prompts/')