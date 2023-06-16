/// Computes LogLog-Beta estimate bias correction using Horner's method.
///
/// Paper: https://arxiv.org/pdf/1612.02284.pdf
/// Wikipedia: https://en.wikipedia.org/wiki/Horner%27s_method
pub(crate) fn beta_horner(z: f64, precision: usize) -> f64 {
    let beta = BETA[precision - 4];
    let zl = (z + 1.0).ln();
    let mut res = 0.0;
    for i in (1..8).rev() {
        res = res * zl + beta[i];
    }
    res * zl + beta[0] * z
}

/// LogLog-Beta polynomial coefficients for precision in [4..18] range.
static BETA: [[f64; 8]; 15] = [
    // p = 4
    [
        -0.582581413904517,
        -1.93530035756005,
        11.079323758035073,
        -22.131357446444323,
        22.505391846630037,
        -12.000723834917984,
        3.220579408194167,
        -0.342225302271235,
    ],
    // p = 5
    [
        -0.7518999460733967,
        -0.959003007774876,
        5.59973713221416,
        -8.209763699976552,
        6.509125489447204,
        -2.683029373432373,
        0.5612891113138221,
        -0.0463331622196545,
    ],
    // p = 6
    [
        29.825790096961963,
        -31.328708333772592,
        -10.594252303658228,
        -11.572012568909962,
        3.818875437390749,
        -2.416013032853081,
        0.4542208940970826,
        -0.0575155452020420,
    ],
    // p = 7
    [
        2.810292129082006,
        -3.9780498518175995,
        1.3162680041351582,
        -3.92524863358059,
        2.008083575394647,
        -0.7527151937556955,
        0.1265569894242751,
        -0.0109946438726240,
    ],
    // p = 8
    [
        1.0063354488755052,
        -2.005806664051124,
        1.6436974936651412,
        -2.7056080994056617,
        1.392099802442226,
        -0.4647037427218319,
        0.07384282377269775,
        -0.00578554885254223,
    ],
    // p = 9
    [
        -0.09415657458167959,
        -0.7813097592455053,
        1.7151494675071246,
        -1.7371125040651634,
        0.8644150848904892,
        -0.23819027465047218,
        0.03343448400269076,
        -0.00207858528178157,
    ],
    // p = 10
    [
        -0.25935400670790054,
        -0.5259830199980581,
        1.4893303492587684,
        -1.2964271408499357,
        0.6228475621722162,
        -0.1567232677025104,
        0.02054415903878563,
        -0.00112488483925502,
    ],
    // p = 11
    [
        -4.32325553856025e-01,
        -1.08450736399632e-01,
        6.09156550741120e-01,
        -1.65687801845180e-02,
        -7.95829341087617e-02,
        4.71830602102918e-02,
        -7.81372902346934e03,
        5.84268708489995e-04,
    ],
    // p = 12
    [
        -3.84979202588598e-01,
        1.83162233114364e-01,
        1.30396688841854e-01,
        7.04838927629266e-02,
        -8.95893971464453e-03,
        1.13010036741605e-02,
        -1.94285569591290e-03,
        2.25435774024964e-04,
    ],
    // p = 13
    [
        -0.41655270946462997,
        -0.22146677040685156,
        0.38862131236999947,
        0.4534097974606237,
        -0.36264738324476375,
        0.12304650053558529,
        -0.0170154038455551,
        0.00102750367080838,
    ],
    // p = 14
    [
        -3.71009760230692e-01,
        9.78811941207509e-03,
        1.85796293324165e-01,
        2.03015527328432e-01,
        -1.16710521803686e-01,
        4.31106699492820e-02,
        -5.99583540511831e-03,
        4.49704299509437e-04,
    ],
    // p = 15
    [
        -0.38215145543875273,
        -0.8906940053609084,
        0.3760233577467887,
        0.9933597744068238,
        -0.6557744163831896,
        0.1833234212970361,
        -0.02241529633062872,
        0.00121399789330194,
    ],
    // p = 16
    [
        -0.3733187664375306,
        -1.41704077448123,
        0.40729184796612533,
        1.5615203390658416,
        -0.9924223353428613,
        0.2606468139948309,
        -0.03053811369682807,
        0.00155770210179105,
    ],
    // p = 17
    [
        -0.36775502299404605,
        0.5383142235137797,
        0.7697028927876792,
        0.5500258358645056,
        -0.7457558826114694,
        0.2571183578582195,
        -0.03437902606864149,
        0.00185949146371616,
    ],
    // p = 18
    [
        -0.3647962332596054,
        0.9973041232863503,
        1.5535438623008122,
        1.2593267719802892,
        -1.5332594820911016,
        0.4780104220005659,
        -0.05951025172951174,
        0.00291076804642205,
    ],
];
