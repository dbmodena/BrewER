<h1>Altosight Dataset</h1>

Here you can find the original raw version of the dataset about USB sticks provided by <a href="https://altosight.com">Altosight</a>, whose subset was used in the <a href="https://dbgroup.ing.unimo.it/sigmod21contest">SIGMOD 2021 Programming Contest</a>.

Note that the id represents the unique identifier of the entity the record refers to (i.e., it identifies the cluster of matching records).
The description of the product can appear in multiple languages and the price is expressed in different currencies.

If you find this dataset useful for your research, please consider citing our <a href="https://doi.org/10.14778/3523210.3523226">paper</a>:

    @article{brewer,
      author    = {Giovanni {Simonini} and Luca {Zecchini} and Sonia {Bergamaschi} and Felix {Naumann}},
      title     = {{Entity Resolution On-Demand}},
      journal   = {{Proceedings of the VLDB Endowment (PVLDB)}},
      volume    = {15},
      number    = {7},
      pages     = {1506--1518},
      year      = {2022},
      doi       = {10.14778/3523210.3523226}
    }

In the dedicated <a href="https://github.com/dbmodena/BrewER/tree/main/altosight_dataset/sigmod_2021_programming_contest">folder</a>, you can find the subsets used for the contest (X and Y datasets were made available on the contest website, while Z and E datasets were used for the evaluation).

In this case, you can reference our report:

    @article{sigmodcontests,
      author    = {Andrea {De Angelis} and Maurizio {Mazzei} and Federico {Piai} and Paolo {Merialdo} and Giovanni {Simonini} and Luca {Zecchini} and Sonia {Bergamaschi} and Donatella {Firmani} and Xu {Chu} and  Peng {Li} and  Renzhi {Wu}},
      title     = {{Experiences and Lessons Learned from the SIGMOD Entity Resolution Programming Contests}},
      journal   = {{SIGMOD Record}},
      volume    = {52},
      number    = {2},
      pages     = {43--47},
      year      = {2023},
      doi       = {10.1145/3615952.3615965}
    }
