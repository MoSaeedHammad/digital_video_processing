TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS +=  /O2 /arch:AVX512



SOURCES += \
        image_utils.cpp \
        main.cpp

HEADERS += \
    image_utils.h
