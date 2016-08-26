#include "l2cHello.h"

extern "C" 
{
    Hello* Hello_new(){
        return new Hello;
    }

    const char* Hello_World(Hello* self){
        return self->World();
    }

    int Hello_ShowImage(Hello* self,const char* fpath){
        return self->ShowImage(fpath);
    }

    void Hello_gc(Hello* self) {
        delete self;
    }
}

