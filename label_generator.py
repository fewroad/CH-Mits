import os
import csv

# 定义数据目录
positive_dir = 'p'
negative_dir = 'n'

# 打印目录内容，检查是否存在文件
print(f"Positive directory: {positive_dir}")
print(f"Files in positive directory: {os.listdir(positive_dir)[:10]}...")  # 只打印前10个
print(f"Negative directory: {negative_dir}")
print(f"Files in negative directory: {os.listdir(negative_dir)[:10]}...")  # 只打印前10个

# 创建label.csv文件
with open('label.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['sentences', 'labels'])
    
    # 处理正面样本
    positive_count = 0
    for folder in os.listdir(positive_dir):
        folder_path = os.path.join(positive_dir, folder)
        if os.path.isdir(folder_path):
            detail_file = os.path.join(folder_path, 'detail.txt')
            if os.path.exists(detail_file):
                try:
                    with open(detail_file, 'r', encoding='utf-8') as df:
                        content = df.read()
                        # 提取文本内容（可以根据实际格式进行处理）
                        writer.writerow([content, 1])
                        positive_count += 1
                except UnicodeDecodeError:
                    try:
                        with open(detail_file, 'r', encoding='gbk') as df:
                            content = df.read()
                            # 提取文本内容（可以根据实际格式进行处理）
                            writer.writerow([content, 1])
                            positive_count += 1
                    except UnicodeDecodeError:
                        # 如果两种编码都失败，使用errors='replace'处理
                        with open(detail_file, 'r', encoding='utf-8', errors='replace') as df:
                            content = df.read()
                            # 提取文本内容（可以根据实际格式进行处理）
                            writer.writerow([content, 1])
                            positive_count += 1
    
    # 处理负面样本
    negative_count = 0
    for folder in os.listdir(negative_dir):
        folder_path = os.path.join(negative_dir, folder)
        if os.path.isdir(folder_path):
            detail_file = os.path.join(folder_path, 'detail.txt')
            if os.path.exists(detail_file):
                try:
                    with open(detail_file, 'r', encoding='utf-8') as df:
                        content = df.read()
                        # 提取文本内容（可以根据实际格式进行处理）
                        writer.writerow([content, 0])
                        negative_count += 1
                except UnicodeDecodeError:
                    try:
                        with open(detail_file, 'r', encoding='gbk') as df:
                            content = df.read()
                            # 提取文本内容（可以根据实际格式进行处理）
                            writer.writerow([content, 0])
                            negative_count += 1
                    except UnicodeDecodeError:
                        # 如果两种编码都失败，使用errors='replace'处理
                        with open(detail_file, 'r', encoding='utf-8', errors='replace') as df:
                            content = df.read()
                            # 提取文本内容（可以根据实际格式进行处理）
                            writer.writerow([content, 0])
                            negative_count += 1
    
    print(f"Processed {positive_count} positive samples and {negative_count} negative samples")
    if positive_count == 0 and negative_count == 0:
        print("Warning: No samples processed! Check if the directories exist and contain detail.txt files.")
