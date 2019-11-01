CC=mpiCC
CFLAGS=-c -g -Wall -std=c++11 -fopenmp
LDFLAGS=-lm -fopenmp
OBJECTS=$(SOURCES:.cpp=.o)

SOURCES=main.cpp savebmp.c

EXECUTABLE=project.x

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o *.x *.bmp *.jpg
