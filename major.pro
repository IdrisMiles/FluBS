
QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Major
DESTDIR = ./bin
TEMPLATE = app


SOURCES += $$PWD/src/*.cpp
HEADERS  += $$PWD/include/*.h
OTHER_FILES += shader/*

INCLUDEPATH +=  $$PWD/include \
                /usr/local/include \
                /usr/include \
                /home/idris/dev/include

LIBS += -L/usr/local/lib -L/usr/lib -lGL -lGLU -lGLEW \
        -L${HOME}/dev/lib -L/usr/local/lib -lassimp

OBJECTS_DIR = ./obj
MOC_DIR = ./moc
FORMS    += ./form/mainwindow.ui

UI_DIR += ./ui


#--------------------------------------------------------------------------
# CUDA stuff
#--------------------------------------------------------------------------

# Make cuda files appear in project
#OTHER_FILES += $$PWD/cuda_src/*.cu
HEADERS += $$PWD/cuda_inc/*.cuh
INCLUDEPATH += $$PWD/cuda_inc
CUDA_SOURCES += $$PWD/cuda_src/*.cu
CUDA_PATH = /usr
NVCC = $$CUDA_PATH/bin/nvcc -ccbin g++

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
GENCODE_FLAGS += -arch=sm_50
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $(CUDA_PATH)/include $(CUDA_PATH)/include/cuda #$(CUDA_PATH)/share/doc/cuda-samples-7.5.18/common/inc

# library directories
QMAKE_LIBDIR += $$CUDA_PATH/lib/x86_64-linux-gnu $(CUDA_PATH)/include/cuda

CUDA_OBJECTS_DIR = $$PWD/cuda_obj

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,' -I','-I','')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += -lcudart -lcurand

cuda.input = CUDA_SOURCES
cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
cuda.commands = $$NVCC -m$$SYSTEM_TYPE $$GENCODE_FLAGS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} $$NVCC_OPTIONS $$CUDA_INC
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda
