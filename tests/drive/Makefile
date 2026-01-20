CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99 -I../../
TARGET = test_error
SOURCE = test_error.c

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: test clean
