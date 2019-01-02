.PHONY: clean all

##CXXFLAGS += -W -Wall -Wextra -Werror -std=c++11
CXXFLAGS += -std=gnu++11

all: clean
	g++ -O2 -DNDEBUG main.cpp NNetwork.cpp -v -o nn.out

gcc49:
	g++-4.9 -std=c++11 -O2 -DNDEBUG main.cpp NNetwork.cpp -v -o nn49.out

clean:
	@rm -rf nn.out

