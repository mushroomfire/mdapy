# Generic GNUMakefile

# Just a snippet to stop executing under other make(1) commands
# that won't understand these lines
ifneq (,)
This makefile requires GNU Make.
endif

PROGRAM = benchmark
CPP_FILES = main.cpp unittest.cpp\
	ptm_alloy_types.cpp\
	ptm_canonical_coloured.cpp \
	ptm_convex_hull_incremental.cpp \
	ptm_deformation_gradient.cpp \
	ptm_graph_data.cpp\
	ptm_graph_tools.cpp \
	ptm_index.cpp\
	ptm_initialize_data.cpp \
	ptm_multishell.cpp\
	ptm_neighbour_ordering.cpp\
	ptm_normalize_vertices.cpp \
	ptm_polar.cpp \
	ptm_quat.cpp \
	ptm_structure_matcher.cpp \
	ptm_voronoi_cell.cpp

#COBJS := $(patsubst %.c, %.o, $(C_FILES))
CPPOBJS := $(patsubst %.cpp, %.o, $(CPP_FILES))
LDFLAGS =
LDLIBS = -lm #-fno-omit-frame-pointer -fsanitize=address

#CC = gcc
CPP = g++

HEADER_FILES = ptm_alloy_types.h\
	ptm_canonical_coloured.h \
	ptm_convex_hull_incremental.h \
	ptm_deformation_gradient.h\
	ptm_fundamental_mappings.h \
	ptm_graph_data.h\
	ptm_graph_tools.h \
	ptm_index.h \
	ptm_initialize_data.h \
	ptm_multishell.h\
	ptm_neighbour_ordering.h \
	ptm_normalize_vertices.h \
	ptm_polar.h \
	ptm_quat.h \
	ptm_structure_matcher.h \
	ptm_voronoi_cell.h

OBJDIR = .

#C_OBJECT_FILES = $(C_SRC_FILES:%.c=$(OBJDIR)/%.o) 
CPP_OBJECT_FILES = $(CPP_SRC_FILES:%.cpp=$(OBJDIR)/%.o) 
C_OBJECT_MODULE_FILE = $(C_SRC_MODULE_FILE:%.c=$(OBJDIR)/%.o) 

#CFLAGS = -std=c99 -g -O3 -Wall -Wextra
CPPFLAGS = -g -O3 -std=c++11 -Wall -Wextra -Wvla -pedantic #-fno-omit-frame-pointer -fsanitize=address


all: $(PROGRAM)

#$(PROGRAM): $(COBJS) $(CPPOBJS)
#	$(CC) -c $(CFLAGS) $(COBJS)
#	$(CPP) -c $(CPPFLAGS) $(CPPOBJS)
#	$(CPP) $(COBJS) $(CPPOBJS) -o $(PROGRAM) $(LDLIBS) $(LDFLAGS)

$(PROGRAM): $(CPPOBJS)
	$(CPP) -c $(CPPFLAGS) $(CPPOBJS)
	$(CPP) $(CPPOBJS) -o $(PROGRAM) $(LDLIBS) $(LDFLAGS)

# These are the pattern matching rules. In addition to the automatic
# variables used here, the variable $* that matches whatever % stands for
# can be useful in special cases.
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%: %.c
	$(CC) $(CFLAGS) -o $@ $<


