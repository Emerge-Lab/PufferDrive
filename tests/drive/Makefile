CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99 -I../../ -I../../raylib-5.5_macos/include
TARGET = test_error
SOURCE = test_error.c
COLLISION_TARGET = test_collision_responsibility
COLLISION_SOURCE = test_collision_responsibility.c

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

$(COLLISION_TARGET): $(COLLISION_SOURCE)
	$(CC) $(CFLAGS) -o $(COLLISION_TARGET) $(COLLISION_SOURCE) ../../raylib-5.5_macos/lib/libraylib.a -lm -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

test: $(TARGET)
	./$(TARGET)

collision-responsibility: $(COLLISION_TARGET)
	./$(COLLISION_TARGET)

clean:
	rm -f $(TARGET) $(COLLISION_TARGET)

.PHONY: test collision-responsibility clean
