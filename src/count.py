def count_csv_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for _ in file)
    return line_count

if __name__ == "__main__":
    file_path = "D:/Eclipse/C_BIRT/BIRT_dataset_issues.csv" # Cambia esto con la ruta de tu archivo
    print(f"El archivo tiene {count_csv_lines(file_path)} líneas.")