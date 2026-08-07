

# MSLTNet
WACV 2024 (Implementación oficial de "4K-Resolution Photo Exposure Correction at 125 FPS with ~ 8K Parameters") [[arxiv](https://arxiv.org/abs/2311.08759)] [[Paper](https://openaccess.thecvf.com/content/WACV2024/html/Zhou_4K-Resolution_Photo_Exposure_Correction_at_125_FPS_With_8K_Parameters_WACV_2024_paper.html)]

## 
#### Una visión rápida de la comparación en el conjunto de datos ME:
<div align="left">
<img src="https://github.com/Zhou-Yijie/MSLTNet/blob/main/fig1.jpg" height="150px" ><img src="https://github.com/Zhou-Yijie/MSLTNet/blob/main/fig2.jpg" height="150px" >
</div>


## Dependencias e Instalación
```
# create new anaconda env
conda create -n mslt python=3.7 -y
conda activate mslt

# install python dependencies
pip3 install -r requirements.txt
```
## Preparar el Conjunto de Datos
Descargue el Conjunto de Datos ME desde [[Baidu Disk](https://pan.baidu.com/s/1tgVVLpZnUsm1pgi8Df63EQ?pwd=m7ai)] o [[google drive](https://drive.google.com/file/d/1dFqOrIdkPZ2seYRN-Hldl8B7tGWNurlM/view?usp=drive_link)] , descomprima el archivo y colóquelo en data/ 
## Inferencia Rápida
```
# Inference On ME Dataset
python SingleTest.py
```
## Prueba del Rendimiento del Modelo
```
python test_model.py
```
## Entrenar MSLT
```
python Train.py
```
## Citación
Si este trabajo ha sido útil para su investigación, por favor considere citarlo:
```
@inproceedings{zhou2024mslt,
  title={4K-Resolution Photo Exposure Correction at 125 FPS with ~8K Parameters},
  author={Zhou, Yijie and Li, Chao and Liang, Jin and Xu, Tianyi and Liu, Xin and Xu, Jun},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```
## Contacto
Si tiene alguna pregunta, no dude en contactarme a ```yjzhou@mail.nankai.edu.cn```
