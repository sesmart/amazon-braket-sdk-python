# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import functools
import itertools
import math

import numpy as np
import pytest

from braket.circuits import gates
from braket.circuits.circuit import Circuit
from braket.circuits.observables import I, X, Y, Z, Sum
from braket.quantum_information import PauliString

ORDER = ["I", "X", "Y", "Z"]
PAULI_INDEX_MATRICES = {
    0: gates.I().to_matrix(),
    1: gates.X().to_matrix(),
    2: gates.Y().to_matrix(),
    3: gates.Z().to_matrix(),
}
SIGN_MAP = {"+": 1, "-": -1}


@pytest.mark.parametrize(
    "pauli_string, string, phase, observable, obs_with_iden",
    [
        ("+XZ", "XZ", 0, X() @ Z(), X() @ Z()),
        ("-ZXY", "ZXY", math.pi, Z() @ X() @ Y(), Z() @ X() @ Y()),
        ("YIX", "YIX", 0, Y() @ X(), Y() @ I() @ X()),
        (PauliString("-ZYXI"), "ZYXI", math.pi, Z() @ Y() @ X(), Z() @ Y() @ X() @ I()),
        ("IIXIIIYI", "IIXIIIYI", 0, X() @ Y(), I() @ I() @ X() @ I() @ I() @ I() @ Y() @ I()),
    ],
)
def test_happy_case(pauli_string, string, phase, observable, obs_with_iden):
    instance = PauliString(pauli_string)
    assert str(instance) == string
    assert instance.phase == phase
    pauli_list = list(str(pauli_string))
    if pauli_list[0] in {"-", "+"}:
        pauli_list.pop(0)
    stripped = "".join(pauli_list)
    assert len(instance) == len(stripped)
    assert len(instance) == instance.qubit_count
    for i in range(len(instance)):
        assert ORDER[instance[i]] == stripped[i]
    assert instance == PauliString(pauli_string)
    assert instance == PauliString(instance)
    assert instance.to_unsigned_observable() == observable
    assert instance.to_unsigned_observable(include_trivial=True) == obs_with_iden


@pytest.mark.parametrize(
    "other", [PauliString("+XYZ"), PauliString("-XI"), PauliString("-XYZI")]
)
def test_not_equal(other):
    assert PauliString("-XYZ") != other

@pytest.mark.parametrize(
    "other", [None, 1, 1j])
@pytest.mark.xfail(raises=TypeError)
def test_failure_to_launch(other):
    PauliString(other)

@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("invalid_string", ["XAY", "-BYZ", "+", "-", "xyz", ""])
def test_invalid_string(invalid_string):
    PauliString(invalid_string)


@pytest.mark.xfail(raises=TypeError)
def test_invalid_type():
    PauliString(1234)


@pytest.mark.xfail(raises=IndexError)
@pytest.mark.parametrize("string", ["XZY", "-YIIXZ", "+IXIYIZ"])
def test_index_out_of_bounds(string):
    PauliString(string)[len(string)]


@pytest.mark.parametrize(
    "string,weight",
    # Make phase explicit for test simplicity
    list(itertools.product(["-ZYX", "-IXIIXYZ", "+ZXYXY"], [1, 2, 3])),
)
def test_weight_n_substrings(string, weight):
    pauli_string = PauliString(string)
    qubit_count = pauli_string.qubit_count
    nontrivial = [qubit for qubit in range(qubit_count) if pauli_string[qubit]]
    substrings = []
    for indices in itertools.combinations(nontrivial, weight):
        factors = "".join([string[qubit + 1] if qubit in indices else "I" for qubit in range(qubit_count)])
        substrings.append(PauliString(factors))
    actual = pauli_string.weight_n_substrings(weight)
    assert actual == tuple(substrings)
    assert len(actual) == math.comb(len(nontrivial), weight)


@pytest.mark.parametrize(
    "string,signs",
    list(itertools.product(["X", "Y", "Z"], ["+", "-"]))
    + [("ZIY", "+--"), ("YIXIZ", [1, 1, -1, -1, 1]), ("XYZ", (-1, -1, -1)), ("XZIY", None)],
)
def test_eigenstate(string, signs):
    pauli_string = PauliString(string)
    circuit = pauli_string.eigenstate(signs)
    for qubit in range(len(string)):
        circuit.i(qubit)
    initial_state = np.zeros(2 ** len(pauli_string))
    initial_state[0] = 1
    state = circuit.to_unitary() @ initial_state

    if not signs:
        signs = [1] * len(string)
    signs_list = [SIGN_MAP[sign] for sign in signs] if isinstance(signs, str) else signs
    positive = [signs_list[i] for i in range(len(string)) if signs_list[i] < 0 and string[i] != "I"]
    total_sign = 1 if len(positive) % 2 == 0 else -1
    pauli_matrix = functools.reduce(
        np.kron, [PAULI_INDEX_MATRICES[pauli] for pauli in pauli_string]
    )
    actual = total_sign * pauli_matrix @ state
    assert np.allclose(actual, state)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("sign", ["+ab", "--+-", [+2, -1, -1], (-1, 1j, 1)])
def test_eigenstate_invalid_signs(sign):
    PauliString("XYZ").eigenstate(sign)


@pytest.mark.parametrize(
    "circ_arg_1, circ_arg_2, circ_res",
    [
        ("III", "+III", "III"),
        ("Z", "I", "Z"),
        ("I", "-Y", "-Y"),
        ("XYXZY", "+XYXZY", "IIIII"),
        ("XYZ", "ZYX", "YIY"),
        ("YZ", "ZX", "-XY"),
        ("-Z", "Y", "1jX"),
        ("Z", "Y", "-1jX"),
    ],
)
def test_dot(circ_arg_1, circ_arg_2, circ_res):
    circ1 = PauliString(circ_arg_1)
    circ2 = circ1.dot(PauliString(circ_arg_2))
    assert circ2 == PauliString(circ_res)
    assert circ1 == circ1

    # Test in-place computation
    circ1 = PauliString(circ_arg_1)
    circ1.dot(PauliString(circ_arg_2), inplace=True)
    assert circ1 == PauliString(circ_res)

    # Test operator overloads
    circ1 = PauliString(circ_arg_1)
    circ2 = PauliString(circ_arg_2)
    assert circ1 * circ2 == PauliString(circ_res)
    circ1 *= circ2
    assert circ1 == PauliString(circ_res)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize(
    "circ1, circ2, operation",
    [
        (PauliString("III"), PauliString("II"), "dot()"),
        (PauliString("III"), PauliString("II"), "*"),
        (PauliString("III"), PauliString("II"), "*="),
        (PauliString("IXI"), PauliString("II"), "dot()"),
        (PauliString("IXI"), PauliString("II"), "*"),
        (PauliString("IXI"), PauliString("II"), "*="),
    ],
)
def test_dot_unequal_lengths(circ1, circ2, operation):
    if operation == "dot()":
        circ1.dot(circ2)
    elif operation == "*":
        circ1 * circ2
    elif operation == "*=":
        circ1 *= circ2


@pytest.mark.parametrize(
    "circ, n, circ_res",
    [
        ("-X", 1, "-X"),
        ("Y", 2, "I"),
        ("Y", -2, "I"),
        ("-X", 3, "-X"),
        ("XYZ", 5, "XYZ"),
        ("XYZ", -5, "XYZ"),
        ("XYZ", 6, "III"),
        ("XYZ", 1, "XYZ"),
        ("-YX", 0, "II"),
        ("Y", 0, "I"),
    ],
)
def test_power(circ, n, circ_res):
    circ1 = PauliString(circ)
    circ2 = circ1.power(n)
    assert circ2 == PauliString(circ_res)
    assert circ1 == PauliString(circ)

    # Test in-place computation
    circ1.power(n, inplace=True)
    assert circ1 == PauliString(circ_res)

    # Test operator overloads
    circ1 = PauliString(circ)
    assert (circ1**n) == PauliString(circ_res)
    circ1 **= n
    assert circ1 == PauliString(circ_res)


@pytest.mark.xfail(raises=TypeError)
@pytest.mark.parametrize(
    "circ, n, operation",
    [
        (PauliString("XYZ"), "-1.2", "power()"),
        (PauliString("XYZ"), "-1.2", "**"),
        (PauliString("XYZ"), "-1.2", "**="),
        (PauliString("ZYX"), "1.2", "power()"),
        (PauliString("ZYX"), "1.2", "**"),
        (PauliString("ZYX"), "1.2", "**="),
        (PauliString("I"), "3j", "power()"),
        (PauliString("I"), "3j", "**"),
        (PauliString("I"), "3j", "**="),
    ],
)
def test_power_invalid_exp(circ, n, operation):
    if operation == "power()":
        circ.power(n)
    elif operation == "**":
        circ**n
    elif operation == "**=":
        circ **= n


@pytest.mark.parametrize(
    "circ, circ_res",
    [
        (PauliString("I"), Circuit().i(0)),
        (PauliString("-X"), Circuit().x(0)),
        (PauliString("IYX"), Circuit().i(0).y(1).x(2)),
        (PauliString("ZIIIX"), Circuit().z(0).i(1).i(2).i(3).x(4)),
    ],
)
def test_to_circuit(circ, circ_res):
    assert circ.to_circuit() == circ_res

# Test coefficient parsing in string input
@pytest.mark.parametrize(
    "input_str, expected_coeff",
    [
        ("0.523jXYZ", 0.523j),
        ("2.5 XY", 2.5),
        ("-1.5e-2 ZI", -0.015),
        ("1e3jII", 1000j),
        ("2+3jXZ", 2+3j),
    ],
)

def test_coefficient_parsing(input_str, expected_coeff):
    ps = PauliString(input_str)
    assert abs(ps.coeff - expected_coeff) < 1e-10


@pytest.mark.parametrize(
    "pauli_string, expected_observable",
    [
        ("+XZ", X() @ Z()),
        ("-XZ", -1 * (X() @ Z())),
        ("0.5XY", 0.5 * (X() @ Y())),
        ("2jZ", 2j * Z()),
        ("-0.5jYX", -0.5j * (Y() @ X())),
        ("IXIY", X() @ Y()),
        ("2IXIY", 2 * (X() @ Y())),
    ],
)
def test_to_observable(pauli_string, expected_observable):
    ps = PauliString(pauli_string)
    obs = ps.to_observable()
    assert obs == expected_observable


@pytest.mark.parametrize(
    "coeff, pauli_string, expected_result",
    [
        (2, "XY", "2XY"),
        (0.5, "Z", "0.5Z"),
        (1j, "YX", "1jYX"),
        (-1.5, "XZ", "-1.5XZ"),
        (2+3j, "I", "(2+3j)I"),
    ],
)
def test_coefficient_multiplication(coeff, pauli_string, expected_result):
    ps = PauliString(pauli_string)
    
    # Test c * PauliString
    result1 = coeff * ps
    assert isinstance(result1, PauliString)
    assert abs(result1.coeff - coeff) < 1e-10
    
    # Test PauliString * c
    result2 = ps * coeff
    assert isinstance(result2, PauliString)
    assert abs(result2.coeff - coeff) < 1e-10
    
    # Test PauliString *= c
    ps_copy = PauliString(pauli_string)
    ps_copy *= coeff
    assert isinstance(ps_copy, PauliString)
    assert abs(ps_copy.coeff - coeff) < 1e-10
    
    # All should be equivalent
    assert result1 == result2 == ps_copy


@pytest.mark.parametrize(
    "observable, expected_pauli_string",
    [
        (X() @ Z(), "XZ"),
        (Y() @ X(), "YX"),
        (0.5 * (X() @ Y()), "XY"),
        (2j * Z(), "Z"),
        (1 * Z(3), "IIIZ")
    ],
)
def test_from_observable(observable, expected_pauli_string):
    print(observable)
    ps = PauliString.from_observable(observable)
    print(ps.__str__())
    assert str(ps) == expected_pauli_string
    assert abs(ps.coeff - observable.coefficient) < 1e-10


# Test tuple input constructor
@pytest.mark.parametrize(
    "qubit_count, nontrivial, coeff, expected_str",
    [
        (3, {0: "X", 2: "Z"}, 1, "XIZ"),
        (2, {1: "Y"}, -1, "IY"),
        (4, {0: "Z", 1: "X", 3: "Y"}, 0.5j, "ZXIY"),
    ],
)
def test_tuple_constructor(qubit_count, nontrivial, coeff, expected_str):
    ps = PauliString((qubit_count, nontrivial), coeff)
    assert ps.qubit_count == qubit_count
    assert ps._nontrivial == nontrivial
    # Check string representation matches expected pattern
    factors = [nontrivial.get(i, "I") for i in range(qubit_count)]
    assert "".join(factors) == expected_str

# Test polar coordinate coefficient input
def test_polar_coefficient():
    import math
    # Test with polar coordinates (modulus, phase)
    ps = PauliString("XYZ", (2.0, math.pi / 2))  # Should be 2j
    assert abs(ps.coeff - 2j) < 1e-10
    assert ps.modulus == 2.0
    assert abs(ps.phase - math.pi / 2) < 1e-10

# Test match statement cases in _split_pauli_string
@pytest.mark.parametrize(
    "input_str, input_coeff, expected_coeff",
    [
        ("-XYZ", 0.5, -0.5),
        ("-XYZ", -1, +1),
        ("XYZ", 1.5, 1.5),
        ("2.0jXY", -2.0j, 4.0),
    ],
)
def test_split_match_cases_composite(input_str, input_coeff, expected_coeff):
    if expected_coeff == ValueError:
        with pytest.raises(ValueError):
            PauliString(input_str, coeff=input_coeff)
    else:
        ps = PauliString(input_str, coeff=input_coeff)
        assert abs(ps.coeff - expected_coeff) < 1e-10




