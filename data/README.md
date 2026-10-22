# Data (not redistributed)

All datasets are public real-world sensor data; none of them is synthetic. Download them and place the files as below.

| File | Source | SHA-256 |
|---|---|---|
| `data/metr_la.h5`, `data/adj_mx.pkl` | METR-LA and its adjacency matrix from the DCRNN repository (Li et al., ICLR 2018), https://github.com/liyaguang/DCRNN | `e00a068c…0990`, `a35687c6…1b3` |
| `data/PEMS-BAY.csv`, `data/adj_mx_PEMS-BAY.pkl` | PEMS-BAY, Zenodo record 5724362 (time series) and the DCRNN repository (adjacency) | `c39dc8f5…1ef2`, `116275f5…a370` |
| `data/pemsd7m/PeMSD7_V_228.csv`, `PeMSD7_W_228.csv` | PeMSD7(M) from the STGCN repository (Yu et al., IJCAI 2018), https://github.com/VeritasYin/STGCN_IJCAI-18 | `9731c987…8b3`, `c558c602…0060e` |

Full hashes of the files used in the paper:

```
e00a068ca5ec0abcf225b5a044bf962cc1c32c4fef31bad8187c04f342ef0990  data/metr_la.h5
a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3  data/adj_mx.pkl
c39dc8f5f3a7f1e9e9deeab1f3fdfa7b87d2b4e45b4075d4c3bd3c62c3051ef2  data/PEMS-BAY.csv
116275f5704d0d492018e14c047f7ae5004385b450aa907f88424055a4a97370  data/adj_mx_PEMS-BAY.pkl
9731c9874a05b5ca88f482062760efb0cbf38816721969a072907fe381d6adb7  data/pemsd7m/PeMSD7_V_228.csv
c558c602b7937fb3ecb3835938fe8f61a34efa4c51e0ae3f61ed93ecc680060e  data/pemsd7m/PeMSD7_W_228.csv
```
Verify with `sha256sum -c` after saving the list as a file. Please respect the original licences and terms of the data providers.
