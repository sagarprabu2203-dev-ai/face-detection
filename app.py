import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="IVA ASSIGNMENT",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# WHITE + BLACK THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
    color: #000000;
}

.block-container {
    max-width: 850px;
    padding-top: 40px;
    padding-bottom: 50px;
}


/* ALL TITLES BLACK */

h1,
h2,
h3,
h4,
h5,
h6 {
    color: #000000 !important;
}


/* NORMAL TEXT BLACK */

p {
    color: #000000 !important;
}


/* FILE UPLOADER TEXT */

label {
    color: #000000 !important;
}

[data-testid="stFileUploader"] label {
    color: #000000 !important;
}


/* MAIN TITLE */

.main-title {
    color: #000000 !important;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}


/* SUBTITLE */

.main-subtitle {
    color: #000000 !important;
    text-align: center;
    font-size: 14px;
    margin-bottom: 30px;
}


/* SECTION TITLES */

.section-title {
    color: #000000 !important;
    font-size: 28px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 8px;
}


/* SECTION NUMBER */

.section-number {
    color: #000000 !important;
    font-size: 13px;
    font-weight: 700;
}


/* SECTION DESCRIPTION */

.section-description {
    color: #000000 !important;
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 20px;
}


/* BUTTON */

.stButton > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #000000 !important;
    border-radius: 8px !important;
}


/* METRIC */

[data-testid="stMetricLabel"] {
    color: #000000 !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
}


/* ALERT TEXT */

[data-testid="stAlert"] {
    color: #000000 !important;
}


/* DIVIDER */

hr {
    border-color: #000000 !important;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #000000 !important;
    font-size: 11px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">IVA ASSIGNMENT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Computer Vision'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# IMAGE CONVERSION
# ============================================================

def uploaded_to_cv(uploaded_file):

    image_bytes = uploaded_file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_np = np.array(image)

    image_cv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    return image_cv


def cv_to_rgb(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# HAAR CASCADE
# ============================================================

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)


# ============================================================
# 01 — FACE DETECTION
# ============================================================

st.markdown(
    '<div class="section-number">01</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Face Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload an image to detect human faces and analyze '
    'the detected facial regions.'
    '</div>',
    unsafe_allow_html=True
)


face_file = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="face_detection"
)


if face_file is not None:

    image = uploaded_to_cv(
        face_file
    )


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    output = image.copy()


    for (x, y, w, h) in faces:

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )


    st.image(
        cv_to_rgb(output),
        caption="Face Detection Result",
        use_container_width=True
    )


    st.metric(
        "Faces Detected",
        len(faces)
    )


    if len(faces) > 0:

        x, y, w, h = faces[0]


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "X",
                x
            )


        with col2:

            st.metric(
                "Y",
                y
            )


        with col3:

            st.metric(
                "Width",
                w
            )


        with col4:

            st.metric(
                "Height",
                h
            )


    else:

        st.warning(
            "No face detected."
        )


st.divider()


# ============================================================
# FACE VERIFICATION FUNCTIONS
# ============================================================

def get_largest_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    if len(faces) == 0:

        return None


    largest = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )


    x, y, w, h = largest


    return image[
        y:y + h,
        x:x + w
    ]


def prepare_face(face):

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.resize(
        gray,
        (128, 128)
    )


    gray = cv2.equalizeHist(
        gray
    )


    return gray.flatten().astype(
        np.float32
    )


# ============================================================
# 02 — FACE VERIFICATION
# ============================================================

st.markdown(
    '<div class="section-number">02</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Face Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload two face images to compare their similarity '
    'and verification result.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:

    image1_file = st.file_uploader(
        "Upload Image 1",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="verification_image_1"
    )


with col2:

    image2_file = st.file_uploader(
        "Upload Image 2",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="verification_image_2"
    )


if image1_file is not None and image2_file is not None:

    image1 = uploaded_to_cv(
        image1_file
    )


    image2 = uploaded_to_cv(
        image2_file
    )


    face1 = get_largest_face(
        image1
    )


    face2 = get_largest_face(
        image2
    )


    if face1 is None or face2 is None:

        st.error(
            "Face not detected in one or both images."
        )


    else:

        data1 = prepare_face(
            face1
        )


        data2 = prepare_face(
            face2
        )


        correlation = np.corrcoef(
            data1,
            data2
        )[0, 1]


        if np.isnan(correlation):

            correlation = 0.0


        similarity = (
            (correlation + 1.0) / 2.0
        ) * 100


        similarity = max(
            0,
            min(100, similarity)
        )


        if similarity >= 70:

            st.success(
                "MATCH"
            )

        else:

            st.error(
                "NO MATCH"
            )


        st.metric(
            "Similarity",
            f"{similarity:.2f}%"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.image(
                cv_to_rgb(face1),
                caption="Face 1",
                use_container_width=True
            )


        with col2:

            st.image(
                cv_to_rgb(face2),
                caption="Face 2",
                use_container_width=True
            )


st.divider()


# ============================================================
# 03 — PATTERN LOCALIZATION
# ============================================================

st.markdown(
    '<div class="section-number">03</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Pattern Localization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload a main image and a template image to locate '
    'the matching pattern.'
    '</div>',
    unsafe_allow_html=True
)


main_file = st.file_uploader(
    "Upload Main Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="pattern_main"
)


template_file = st.file_uploader(
    "Upload Template Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="pattern_template"
)


if main_file is not None and template_file is not None:

    main_image = uploaded_to_cv(
        main_file
    )


    template_image = uploaded_to_cv(
        template_file
    )


    main_gray = cv2.cvtColor(
        main_image,
        cv2.COLOR_BGR2GRAY
    )


    template_gray = cv2.cvtColor(
        template_image,
        cv2.COLOR_BGR2GRAY
    )


    main_h, main_w = main_gray.shape


    template_h, template_w = template_gray.shape


    if (
        template_h > main_h
        or template_w > main_w
    ):

        st.error(
            "Template image must be smaller than the main image."
        )


    else:

        result = cv2.matchTemplate(
            main_gray,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )


        min_val, max_val, min_loc, max_loc = (
            cv2.minMaxLoc(result)
        )


        x, y = max_loc


        output = main_image.copy()


        cv2.rectangle(
            output,
            (x, y),
            (
                x + template_w,
                y + template_h
            ),
            (255, 255, 255),
            3
        )


        st.image(
            cv_to_rgb(output),
            caption="Pattern Localization Result",
            use_container_width=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Match Score",
                f"{max_val:.2f}"
            )


        with col2:

            st.metric(
                "X Position",
                x
            )


        with col3:

            st.metric(
                "Y Position",
                y
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'IVA ASSIGNMENT • COMPUTER VISION'
    '</div>',
    unsafe_allow_html=True
)