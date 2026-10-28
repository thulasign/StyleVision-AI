
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
import colorsys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="StyleVision AI",
    layout="wide"
)

st.title("StyleVision AI")
st.write(
    "Upload a clothing photo to identify its category "
    "and discover simple styling ideas."
)

st.divider()

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("stylevision_model_v2.keras")

@st.cache_data
def load_class_names():
    with open("stylevision_classes_v2.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

model = load_model()
class_names = load_class_names()

# ---------------- STYLING SUGGESTIONS ----------------

styling_tips = {
    "Casual Shoes": [
        "Pair with jeans or chinos for a relaxed outfit.",
        "Choose colours that complement your clothing.",
        "Keep your shoes clean for a polished look."
    ],
    "Dresses": [
        "Pair with flats for a comfortable everyday look.",
        "Add a handbag and simple accessories.",
        "Try a light jacket when the weather is cooler."
    ],
    "Handbags": [
        "Match your bag with your shoes or accessories.",
        "Try a crossbody bag for everyday outings.",
        "Choose a structured bag for formal occasions."
    ],
    "Sandals": [
        "Pair with summer dresses or casual outfits.",
        "Choose comfortable sandals for everyday use.",
        "Coordinate the colour with your accessories."
    ],
    "Shirts": [
        "Pair with jeans for a casual outfit.",
        "Wear with tailored trousers for a formal look.",
        "Roll up the sleeves for a relaxed style."
    ],
    "Sports Shoes": [
        "Pair with joggers or sportswear.",
        "Choose comfortable clothing for active days.",
        "Coordinate your shoes with your outfit."
    ],
    "Tops": [
        "Pair with jeans, skirts, or trousers.",
        "Layer with a jacket for a different look.",
        "Add simple accessories to complete the outfit."
    ],
    "Trousers": [
        "Pair with a fitted shirt or casual top.",
        "Try sneakers for a casual look or loafers for a formal look.",
        "Choose a top that complements the trouser colour."
    ],
    "Tshirts": [
        "Pair with jeans or denim shorts.",
        "Add sneakers for a casual look.",
        "Layer with a jacket in cooler weather."
    ]
}

# ---------------- IMAGE UPLOAD ----------------
# ---------------- COLOUR MATCHING ----------------

def get_dominant_colours(image, number=3):
    """Find the most common colours in the uploaded image."""
    small_image = image.resize((100, 100))
    pixels = list(small_image.getdata())

    # Reduce similar RGB values into colour groups
    simplified = [
        (r // 32 * 32, g // 32 * 32, b // 32 * 32)
        for r, g, b in pixels
    ]

    common = Counter(simplified).most_common(number)

    return [
        tuple(min(channel + 16, 255) for channel in colour)
        for colour, _ in common
    ]


def describe_colour(rgb):
    """Give an approximate human-readable colour name."""
    r, g, b = rgb

    if max(r, g, b) < 55:
        return "Black"
    if min(r, g, b) > 215:
        return "White"
    if max(r, g, b) - min(r, g, b) < 25:
        if sum(rgb) / 3 > 175:
            return "Light grey"
        if sum(rgb) / 3 > 100:
            return "Grey"
        return "Dark grey"

    if r > 170 and g > 120 and b < 100:
        return "Peach" if r - g < 55 else "Orange"
    if r > 150 and g < 100 and b < 110:
        return "Red"
    if r > 150 and g > 130 and b < 100:
        return "Yellow"
    if g > r * 1.2 and g > b * 1.1:
        return "Green"
    if b > r * 1.15 and b > g * 1.05:
        return "Blue"
    if r > 100 and b > 100 and g < min(r, b) * 0.8:
        return "Purple"
    if r > 100 and g > 65 and b < 70:
        return "Brown"

    return "Mixed colour"


def get_matching_colours(rgb):
    """Suggest outfit colours based on the dominant colour."""
    r, g, b = [value / 255 for value in rgb]
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)

    if value < 0.22:
        return ["White", "Beige", "Light grey"]
    if saturation < 0.15 and value > 0.8:
        return ["Black", "Navy blue", "Burgundy"]
    if saturation < 0.15:
        return ["White", "Cream", "Navy blue"]

    complementary_hue = (hue + 0.5) % 1
    analogous_hue = (hue + 0.08) % 1

    suggestions = []

    for h in [complementary_hue, analogous_hue]:
        rr, gg, bb = colorsys.hsv_to_rgb(h, 0.55, 0.75)
        colour_rgb = (int(rr * 255), int(gg * 255), int(bb * 255))
        suggestions.append(describe_colour(colour_rgb))

    suggestions.extend(["White", "Beige"])
    return list(dict.fromkeys(suggestions))[:4]


# ---------------- OUTFIT RECOMMENDATIONS ----------------

outfit_recommendations = {
    "Dresses": [
        "Pair with flats or comfortable heels.",
        "Add a handbag that complements the dress.",
        "Try a denim jacket for a casual look."
    ],
    "Tops": [
        "Pair with jeans, skirts, or tailored trousers.",
        "Try white sneakers for a casual look.",
        "Add a light jacket for layering."
    ],
    "Tshirts": [
        "Pair with jeans or denim shorts.",
        "Add sneakers and a casual backpack.",
        "Try a checked shirt as an outer layer."
    ],
    "Shirts": [
        "Pair with jeans for a casual outfit.",
        "Wear with tailored trousers for a formal look.",
        "Add loafers or simple sneakers."
    ],
    "Jeans": [
        "Pair with a T-shirt for a relaxed outfit.",
        "Try a shirt and loafers for a smart-casual look.",
        "Add sneakers for everyday wear."
    ],
    "Trousers": [
        "Pair with a shirt or fitted top.",
        "Try loafers for a formal outfit.",
        "Choose a complementary belt and shoes."
    ],
    "Kurtas": [
        "Pair with leggings, straight pants, or palazzos.",
        "Try sandals or juttis.",
        "Add simple earrings or a matching dupatta."
    ],
    "Kurtis": [
        "Pair with leggings or jeans.",
        "Try flats or sandals.",
        "Add a simple handbag."
    ],
    "Casual Shoes": [
        "Pair with jeans and a casual top.",
        "Try chinos for a smart-casual look.",
        "Coordinate your shoes with your accessories."
    ],
    "Sports Shoes": [
        "Pair with joggers or sportswear.",
        "Choose breathable clothing for active days.",
        "Try matching your shoes with a T-shirt."
    ],
    "Handbags": [
        "Coordinate your bag with your shoes.",
        "Try a neutral bag for versatile everyday styling.",
        "Choose a structured bag for formal occasions."
    ]
}


# ---------------- CLOTHING CARE ASSISTANT ----------------

clothing_care = {
    "Dresses": [
        "Check the care label before washing.",
        "Use a gentle cycle for delicate fabrics when permitted.",
        "Air-dry when suitable for the fabric."
    ],
    "Shirts": [
        "Separate whites and colours before washing.",
        "Check the fabric label for ironing temperature.",
        "Hang promptly to reduce wrinkles."
    ],
    "Tshirts": [
        "Wash with similar colours.",
        "Turn printed T-shirts inside out before washing.",
        "Avoid high heat if the care label advises against it."
    ],
    "Jeans": [
        "Wash inside out with similar colours.",
        "Avoid unnecessarily frequent washing.",
        "Follow the label's drying instructions."
    ],
    "Trousers": [
        "Check whether the fabric requires dry cleaning.",
        "Use the recommended washing temperature.",
        "Iron according to the care label."
    ],
    "Kurtas": [
        "Check whether the fabric needs gentle washing.",
        "Wash similar colours together.",
        "Dry according to the garment's care label."
    ],
    "Kurtis": [
        "Check the label before washing embellished garments.",
        "Use a suitable gentle wash if permitted.",
        "Avoid ironing directly over embellishments."
    ],
    "Sweaters": [
        "Check whether handwashing or dry cleaning is required.",
        "Avoid stretching delicate knitwear.",
        "Lay flat to dry when recommended."
    ],
    "Jackets": [
        "Check the label before washing.",
        "Some structured jackets require professional cleaning.",
        "Store dry and follow the garment's care instructions."
    ],
    "Handbags": [
        "Test cleaning products on a hidden area first.",
        "Use cleaning methods appropriate for the material.",
        "Store away from excessive moisture and direct sunlight."
    ],
    "Casual Shoes": [
        "Remove loose dirt before cleaning.",
        "Use a cleaning method suitable for the shoe material.",
        "Allow shoes to dry completely before storing."
    ],
    "Sports Shoes": [
        "Brush away loose dirt after use.",
        "Follow the manufacturer's cleaning instructions.",
        "Air-dry instead of using direct high heat."
    ]
}


st.subheader("📷 Upload a Clothing Image")

uploaded_file = st.file_uploader(
    "Choose a clothing photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.image(
                original_image,
                caption="Your uploaded photo",
                use_container_width=True
            )

        # ---------------- PREPROCESS IMAGE ----------------

        image = original_image.resize((160, 160))

        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # ---------------- PREDICTION ----------------

        prediction = model.predict(img_array, verbose=0)[0]

        top_indices = np.argsort(prediction)[::-1][:min(3, len(class_names))]

        best_index = int(top_indices[0])
        best_label = class_names[best_index]
        best_score = float(prediction[best_index]) * 100

        with right:
            st.subheader("🔍 Recognition Results")

            st.metric(
                label="Predicted category",
                value=best_label,
                delta=f"{best_score:.1f}% model score"
            )

            st.caption(
                "Scores are model outputs, not guaranteed probabilities "
                "of correctness. Results may be less reliable for photos "
                "that differ from the training dataset."
            )

            st.subheader("🏆 Top Predictions")

            for rank, index in enumerate(top_indices, start=1):
                label = class_names[int(index)]
                score = float(prediction[index]) * 100

                st.write(f"**{rank}. {label} — {score:.2f}%**")
                st.progress(float(prediction[index]))

        # ---------------- STYLING IDEAS ----------------

        st.divider()
        st.subheader("✨ Fashion Styling Suggestions")

        tips = styling_tips.get(
            best_label,
            [
                "Choose colours that complement your outfit.",
                "Consider the occasion when selecting accessories.",
                "Prioritise comfort alongside style."
            ]
        )

        for tip in tips:
            st.markdown(f"- {tip}")

        # ---------------- COLOUR MATCHING ----------------

        st.divider()
        st.subheader("🎨 Colour Matching")

        dominant_colours = get_dominant_colours(original_image)

        st.write("Main colours detected in your photo:")

        colour_columns = st.columns(len(dominant_colours))

        for column, colour in zip(colour_columns, dominant_colours):
            colour_name = describe_colour(colour)

            with column:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgb{colour};
                        height: 65px;
                        border-radius: 10px;
                        border: 1px solid #999;
                    "></div>
                    """,
                    unsafe_allow_html=True
                )
                st.caption(colour_name)

        if dominant_colours:
            matching_colours = get_matching_colours(
                dominant_colours[0]
            )

            st.write("Suggested matching colours:")
            st.write(" • ".join(matching_colours))

        # ---------------- OUTFIT RECOMMENDATIONS ----------------

        st.divider()
        st.subheader("👗 Outfit Recommendations")

        recommendations = outfit_recommendations.get(
            best_label,
            [
                "Try pairing this item with neutral colours.",
                "Choose accessories that complement your outfit.",
                "Consider the occasion and your comfort."
            ]
        )

        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")

        # ---------------- CLOTHING CARE ----------------

        st.divider()
        st.subheader("🧺 Clothing Care Assistant")

        care_tips = clothing_care.get(
            best_label,
            [
                "Check the garment's care label before cleaning.",
                "Use cleaning products suitable for the material.",
                "Follow the manufacturer's drying instructions."
            ]
        )

        for care_tip in care_tips:
            st.markdown(f"- {care_tip}")



    except Exception as error:
        st.error("Sorry, we couldn't process this image.")
        st.exception(error)

else:
    st.info("Upload a clothing photo to get started.")

st.divider()
st.caption("StyleVision AI | Clothing Recognition & Styling Assistant")