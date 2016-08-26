ffi = require('ffi')

ffi.cdef[[
    typedef struct Hello Hello;
    Hello* Hello_new();
    const char* Hello_World(Hello*);
    int Hello_ShowImage(Hello* self,const char* fpath);
    void Hello_gc(Hello*);
]]

hello = ffi.load('l2cHello')

hello_index = {
    World = hello.Hello_World,
    ShoImg = hello.Hello_ShowImage
}

hello_mt = ffi.metatype('Hello', {
    __index = hello_index
})

myobj = hello.Hello_new()
myobj:ShoImg("logo.png")

