from file_io import read_file
import processing

file_content, file_path = read_file("demo.txt")


if file_content is None:
    print("No file content to process.")
    exit()


print(f"File content loaded successfully -> '{file_path}'\n{file_content[:150]}...\n")

cleaned_text = processing.process_text(file_content)

print(f"\nProcessed content:\n{cleaned_text[:150]}...\n")

tokens = processing.tokenize(cleaned_text)
word_counts = processing.count_words(tokens)

try:
    n = int(input("Enter the number of top words to display: "))
except ValueError:
    print("Please enter a valid number.")
    exit()

top_words = processing.top_n_words(word_counts, n)

print(f"Top {n} words:")
for word, count in top_words:
    print(f"{word:<15} -> {count}")
