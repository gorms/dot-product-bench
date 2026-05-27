CXX     = g++
CXXFLAGS = -O2 -Wall -Wextra

bench: main.cpp
	$(CXX) $(CXXFLAGS) -o bench main.cpp

clean:
	rm -f bench

.PHONY: clean
