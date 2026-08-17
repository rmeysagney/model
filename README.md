# Helio Support Console

Kendi destek verisiyle eğitilmiş, önceden eğitilmiş LLM kullanmayan Flask tabanlı destek öneri arayüzü.

## Yerelde çalıştırma

```bash
pip install -r requirements-training.txt
python app.py
```

Arayüz: `http://127.0.0.1:5050`

## Model eğitimi

```bash
python train_model.py
python model_quality_checks.py
```

Ham eğitim kayıtları ve değerlendirme çıktıları bu repoda izlenmez. Bu dosyalar kişisel veri içerebileceğinden yalnızca güvenli yerel veya özel depolamada tutulmalıdır.

## Yayınlama

`render.yaml` Flask uygulamasını Render Web Service olarak başlatır. Canlıda Gunicorn kullanılır ve `deployment_assets/model_english_multilingual.joblib` yayın modeli yüklenir. Bu dosya eğitim sırasında üretilir; ham JSONL kayıtları yayın deposuna eklenmez.
