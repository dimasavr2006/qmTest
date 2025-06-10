import random
import os


def is_last_line_of_molecule(line_content: str) -> bool:
    """
    Проверяет, является ли строка последней строкой данных молекулы
    (т.е. содержит ровно два числа).
    """
    parts = line_content.strip().split()
    if len(parts) == 2:
        try:
            float(parts[0])
            float(parts[1])
            return True
        except ValueError:
            return False
    return False


def split_molecule_blocks_strict_defined_with_separator(input_filename="test.txt", separator="\n"):
    """
    Читает файл, где данные каждой молекулы представлены как блок:
    - Начинается со строки "dsgdb9nsd_"
    - Заканчивается строкой, содержащей ровно два числа.
    Все строки между этими маркерами (включительно) составляют один блок.
    Строки вне таких блоков игнорируются.

    Создает файлы с 1/2, 1/4 и 1/8 случайных блоков молекул.
    После каждого блока в выходном файле добавляется строка-разделитель.
    """
    molecule_blocks = []
    current_block_lines = []
    in_molecule_block = False

    try:
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            for line_number, raw_line in enumerate(f_in, 1):
                line_content_stripped = raw_line.strip()

                if line_content_stripped.startswith("dsgdb9nsd_"):
                    if in_molecule_block:
                        print(
                            f"Предупреждение (строка {line_number}): Новый блок '{line_content_stripped[:40]}...' начинается, "
                            f"но предыдущий блок (начатый с '{current_block_lines[0].strip()[:40]}...') "
                            f"не был корректно завершен строкой из двух чисел. "
                            f"Предыдущий накопленный блок отброшен.")

                    current_block_lines = [raw_line]
                    in_molecule_block = True

                elif in_molecule_block:
                    current_block_lines.append(raw_line)

                    if is_last_line_of_molecule(line_content_stripped):
                        molecule_blocks.append("".join(current_block_lines))
                        current_block_lines = []
                        in_molecule_block = False

        if in_molecule_block and current_block_lines:
            print(f"Предупреждение: Файл закончился, но последний блок, начатый с "
                  f"'{current_block_lines[0].strip()[:40]}...' не был завершен строкой из двух чисел. "
                  f"Этот незавершенный блок отброшен.")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_filename}' не найден.")
        return
    except Exception as e:
        print(f"Произошла ошибка при чтении файла '{input_filename}': {e}")
        return

    if not molecule_blocks:
        print(f"Молекулярные блоки (определенные началом 'dsgdb9nsd_' и концом из двух чисел) "
              f"не найдены или не удалось их корректно прочитать из файла '{input_filename}'.")
        return

    total_blocks = len(molecule_blocks)
    print(f"Всего найдено корректно завершенных молекулярных блоков: {total_blocks} в файле '{input_filename}'")

    random.shuffle(molecule_blocks)

    denominators = {
        "half": 2,
        "quarter": 4,
        "eighth": 8
    }

    base_name, ext = os.path.splitext(input_filename)

    for suffix, denominator in denominators.items():
        if total_blocks == 0:
            num_to_select = 0
        else:
            num_to_select = total_blocks // denominator

        if num_to_select == 0 and total_blocks > 0:
            print(
                f"Предупреждение: Для выборки '{suffix}' (1/{denominator}) из '{input_filename}' получилось 0 блоков ({total_blocks} // {denominator} = 0). Файл {base_name}_{suffix}{ext} не будет создан.")
            continue

        selected_blocks = molecule_blocks[:num_to_select]
        output_filename = f"{base_name}_{suffix}{ext}"

        try:
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                for i, block_data in enumerate(selected_blocks):
                    f_out.write(block_data)  # Записываем блок как есть
                    # Добавляем разделитель после каждого блока,
                    # но можно не добавлять после самого последнего блока в файле, если не хотите.
                    # В данном случае, для простоты, добавляем всегда.
                    # Если block_data уже заканчивается на \n (что так и есть),
                    # то separator="\n" добавит еще одну пустую строку.
                    # Если separator пустой "", то ничего не добавится.
                    if separator:  # Добавляем разделитель, если он задан
                        f_out.write(separator)

            if selected_blocks:
                print(
                    f"Создан файл: {output_filename} с {len(selected_blocks)} молекулярными блоками (с разделителями).")
            elif total_blocks == 0:
                print(
                    f"Файл {output_filename} создан пустым, так как в '{input_filename}' не найдено корректных блоков.")
        except Exception as e:
            print(f"Произошла ошибка при записи файла '{output_filename}': {e}")


# --- Пример использования ---
if __name__ == "__main__":

    split_molecule_blocks_strict_defined_with_separator("train.txt", separator="\n")