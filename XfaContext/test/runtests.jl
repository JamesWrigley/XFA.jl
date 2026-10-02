import ReTest: retest
import XfaContext

include("XfaContextTests.jl")

retest(XfaContext, XfaContextTests)
