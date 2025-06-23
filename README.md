# Adapters

## Eng version

### How to run?

* Clone the [RIFE](https://github.com/hzwer/ECCV2022-RIFE) repository and download the required weights (see the repository’s instructions).
* Clone the [TPS](https://github.com/tian-one/tps-inbetween) repository and download the required weights (see the repository’s instructions).
* Clone the [SAIN](https://github.com/none-master/SAIN) repository and download the required weights (see the repository’s instructions).

  * You also need to place the file `infer.py` into the SAIN folder. It’s required to run inference.

Once all repositories are cloned and the weights are downloaded, you can run the `main.py` script. The results will be saved in the `output` folder.

---

`frame1.png` & `frame2.png` – test frames
`adapters_sandbox.ipynb` – sandbox for experiments

## Ru version

### Как запустить?
- Клонируем репозиторий [RIFE](https://github.com/hzwer/ECCV2022-RIFE) и скачиваем нужные веса (см. инструкцию репозитория)
- Клонируем репозиторий [TPS](https://github.com/tian-one/tps-inbetween) и скачиваем нужные веса (см. инструкцию репозитория)
- Клонируем репозиторий [SAIN](https://github.com/none-master/SAIN) и скачиваем нужные веса (см. инструкцию репозитория)
  - Также нужно положить файл `infer.py` в папку SAIN. Он нужен для того чтобы запускать инференс.

Как только все репозитории склонированы и веса загружены, то можно запустить файл `main.py`. В папке `output` должны быть результаты

---
`frame1.png` & `frame2.png` - кадры для теста  
`adapters_sandbox.ipynb` - песочница для тестов
