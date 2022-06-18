## Perceiver AR - Pytorch (wip)

Implementation of Perceiver AR, Deepmind's new long-context attention network based on Perceiver architecture, in Pytorch.

I am building this out of popular demand, not because I believe in the architecture. As someone else puts it succinctly, this is equivalent to an encoder / decoder transformer architecture where the encoder has 0 layers (and the decoder cross attention is restricted to 1 layer)

However, the experimental results they provided are still worthwhile and I'll build it out so students and researchers alike can explore along this avenue.

<a href="https://github.com/google-research/perceiver-ar">Official Jax repository</a>

## Citations

```bibtex
@article{Hawthorne2022GeneralpurposeLA,
    title   = {General-purpose, long-context autoregressive modeling with Perceiver AR},
    author  = {Curtis Hawthorne and Andrew Jaegle and Cătălina Cangea and Sebastian Borgeaud and Charlie Nash and Mateusz Malinowski and Sander Dieleman and Oriol Vinyals and Matthew M. Botvinick and Ian Simon and Hannah R. Sheahan and Neil Zeghidour and Jean-Baptiste Alayrac and Jo{\~a}o Carreira and Jesse Engel},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.07765}
}
```
