from braket.circuits import Circuit


test=  Circuit().h(0).h(1)

test = Circuit().add_verbatim_box(test)
print(test)

new = Circuit().h(1) + test.h(0) 
print(new)

new = Circuit().add_verbatim_box(Circuit().add_verbatim_box(new) + test + new)

print(new)

print(new.verbatim)