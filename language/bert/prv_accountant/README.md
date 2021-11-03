# Privacy Random Variable (PRV) Accountant

A fast algorithm to optimally compose privacy guarantees of differentially private (DP) algorithms to arbitrary accuracy.
Our method is based on the notion of privacy loss random variables to quantify the privacy loss of DP algorithms.
For more details see [[1](https://arxiv.org/abs/2106.02848)].

## Installation

```
pip install prv-accountant
```

## Examples

Getting epsilon estimate directly from the command line.

```
compute-dp-epsilon --sampling-probability 5e-3 --noise-multiplier 0.8 --delta 1e-6 --num-compositions 1000
```

Or, use it in python code

```python
from prv_accountant import Accountant

accountant = Accountant(
	noise_multiplier=0.8,
	sampling_probability=5e-3,
	delta=1e-6,
	eps_error=0.1,
	max_compositions=1000
)

eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=1000)
```

For more examples, have a look in the `notebooks` directory.


## References

[1] Sivakanth Gopi, Yin Tat Lee, Lukas Wutschitz. Numerical Composition of Differential Privacy. arXiv. Preprint posted online June 5, 2021. [arXiv](https://arxiv.org/abs/2106.02848)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
