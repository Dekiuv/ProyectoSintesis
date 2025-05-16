import pandas as pd
import csv
import ast
import json

# -----------------------------
# Función para convertir reviews con comillas simples
# -----------------------------
def convertir_reviews(input_path, output_path):
    print(f"Convirtiendo {input_path} (reviews con comillas simples)...")
    data = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                user_id = obj.get('user_id')
                for review in obj.get('reviews', []):
                    review_flat = {
                        'user_id': user_id,
                        'item_id': review.get('item_id'),
                        'recommend': review.get('recommend'),
                        'review_text': review.get('review'),
                        'posted': review.get('posted'),
                        'helpful': review.get('helpful'),
                        'funny': review.get('funny'),
                        'last_edited': review.get('last_edited'),
                    }
                    data.append(review_flat)
            except Exception as e:
                print(f"❌ Error en línea de reviews: {e}")
                continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"✅ {output_path} generado correctamente con {len(df)} registros.")
    else:
        print(f"⚠️ No se pudo generar {output_path}, ninguna línea válida encontrada.")


# -----------------------------
# Función para convertir items con comillas simples
# -----------------------------
def convertir_items(input_path, output_path):
    print(f"Convirtiendo {input_path} (items con comillas simples)...")
    data = []

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)
                data.append(obj)
            except Exception as e:
                print(f"❌ Error en línea de items: {e}")
                continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"✅ {output_path} generado correctamente con {len(df)} registros.")
    else:
        print(f"⚠️ No se pudo generar {output_path}, ninguna línea válida encontrada.")


# -----------------------------
# Función para convertir bundle_data.json
# -----------------------------
def convertir_bundles(input_path, output_path):
    print(f"Convirtiendo {input_path} (bundles)...")
    data = []

    with open(input_path, 'r') as f:
        for line in f:
            try:
                bundle = ast.literal_eval(line.strip())
                bundle_id = bundle.get('bundle_id')
                bundle_name = bundle.get('bundle_name')
                bundle_url = bundle.get('bundle_url')
                bundle_price = bundle.get('bundle_price')
                bundle_final_price = bundle.get('bundle_final_price')
                bundle_discount = bundle.get('bundle_discount')
                for item in bundle.get('items', []):
                    data.append({
                        'bundle_id': bundle_id,
                        'bundle_name': bundle_name,
                        'bundle_url': bundle_url,
                        'bundle_price': bundle_price,
                        'bundle_final_price': bundle_final_price,
                        'bundle_discount': bundle_discount,
                        'item_id': item.get('item_id'),
                        'item_name': item.get('item_name'),
                        'item_url': item.get('item_url'),
                        'item_discounted_price': item.get('discounted_price'),
                        'item_genre': item.get('genre')
                    })
            except Exception as e:
                print(f"❌ Error en línea de bundle: {e}")
                continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"✅ {output_path} generado correctamente con {len(df)} registros.")
    else:
        print(f"⚠️ No se pudo generar {output_path}, ninguna línea válida encontrada.")


# -----------------------------
# Ejecución de los tres
# -----------------------------
convertir_reviews(
    input_path='australian_user_reviews.json',
    output_path='australian_user_reviews.csv'
)

convertir_items(
    input_path='australian_users_items.json',
    output_path='australian_user_items.csv'
)

convertir_bundles(
    input_path='bundle_data.json',
    output_path='bundle_data.csv'
)
