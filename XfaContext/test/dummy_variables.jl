module DummyVariables

using XfaContext

@Group struct Foo
    bar::Parameter{Int}
end

@Variable function compute(::Foo)
    return 42
end

end
