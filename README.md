# Depth_Anything_3_Motifs

Interactive monocular depth reconstruction of Thanjavur temple bas-relief motifs using Depth Anything 3.
<img width="1539" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/2d7b6c57-e9b7-424f-9745-cb59f46a4ae9" />

<img width="1082" height="686" alt="image" src="https://github.com/user-attachments/assets/6161ae6a-6b0f-4864-a8e7-a61fcde0f2c0" />

## Reproducing paper results

For the Section 6.6 mesh validation table, see [VALIDATION.md](VALIDATION.md).
The 35-panel mesh corpus is released at https://doi.org/10.5281/zenodo.19846595.

## Related datasets and glossary of terms

Refer [MOTIF_GLOSSARY](https://github.com/libishm1/Depth_Anything_3_Motifs_CLI/blob/main/MOTIF_GLOSSARY.md) for more detailed description of motifs in tamil traditions

Dataset associated with the repo -
1) https://doi.org/10.5281/zenodo.19455013 . Thanjavur temple motifs, Tamil Nadu
2) https://doi.org/10.5281/zenodo.19468980 . Kulavilakkamman temple , Kalamangalam, Erode. Tamil Nadu
3) https://doi.org/10.5281/zenodo.19469154 .  Narasimhaswamy Temple, Namakkal. Tamil Nadu

## Usage
\
py motifs_interactive.py --image 1.jpg --model large
\

## Controls
| Key | Action |
|-----|--------|
| p | Preview mesh in Open3D |
| e | Edit parameters |
| d | Show depth map and mask |
| s | Save STL |
| r | Re-run depth inference |
| n | Skip image |
| q | Quit |

---

## License

This repository uses a dual license structure.

| Component | License |
|---|---|
| Motif imagery, site records, iconographic documentation | [CC BY-NC-ND 4.0](LICENSE.md) |
| Pipeline software and code | [MIT](LICENSE.md) |

### Motif Dataset — CC BY-NC-ND 4.0

[![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

The motif photographs and heritage documentation are free to use for **conservation, academic research, and educational purposes** with attribution. Commercial use and redistribution of modified versions are not permitted.

**Required attribution:**
> Motif records: Libish Murugesan
> Source: Kalamangalam Temple, Erode District, Tamil Nadu 
> https://github.com/libishm1/Depth_Anything_3_Motifs_CLI

### Software — MIT

The depth reconstruction pipeline (CLI scripts, configuration, processing code) is free to use, modify, and distribute under the MIT License.

### Citation

```bibtex
@software{murugesan2025motifs,
  author    = {Murugesan, Libish},
  title     = {Depth\_Anything\_3\_Motifs\_CLI: Monocular Depth Reconstruction 
               for Temple Bas-Relief Motifs},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/libishm1/Depth_Anything_3_Motifs_CLI}
}
```

See [LICENSE.md](LICENSE.md) for full terms.
