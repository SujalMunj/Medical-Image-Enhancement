document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = window.localStorage.getItem('API_BASE') || 'http://127.0.0.1:5000';
    const uploadBox = document.querySelector('.upload-box');

    if (uploadBox) {
        uploadBox.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.jpg, .jpeg, .png, .dcm';

            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append("file", file);

                try {
                    // ✅ Step 1: Upload to Flask backend
                    const uploadResponse = await fetch(`${API_BASE}/upload`, {
                        method: "POST",
                        body: formData
                    });

                    const uploadResult = await uploadResponse.json();
                    console.log("Upload Response:", uploadResult);

                    if (uploadResult.error) {
                        alert(`❌ Upload failed: ${uploadResult.error}`);
                        return;
                    }

                    // ✅ Step 2: Create a clean result container
                    const oldCard = document.querySelector('.result-card');
                    if (oldCard) oldCard.remove();

                    const card = document.createElement("div");
                    card.className = "result-card";
                    card.style.cssText = `
                        width: 400px;
                        background: #ffffffcc;
                        backdrop-filter: blur(10px);
                        border-radius: 20px;
                        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
                        margin: 30px auto;
                        padding: 20px;
                        text-align: center;
                        font-family: 'Poppins', sans-serif;
                        transition: all 0.3s ease-in-out;
                    `;

                    card.innerHTML = `
                        <h2 style="color:#333;">🩻 X-ray Uploaded</h2>
                        <img src="${uploadResult.file_url}" width="300" style="border-radius:12px; margin:15px 0;">
                        <p style="color:#555;">Processing AI prediction...</p>
                    `;
                    document.body.appendChild(card);

                    // ✅ Step 3: Send filename to /predict
                    const predictResponse = await fetch(`${API_BASE}/predict`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ filename: uploadResult.filename })
                    });

                    const predictResult = await predictResponse.json();
                    console.log("Predict Response:", predictResult);

                    // ✅ Step 4: Display prediction result beautifully
                    if (predictResult.error) {
                        card.innerHTML += `<p style="color:red;">⚠️ ${predictResult.error}</p>`;
                    } else {
                        card.innerHTML = `
                            <h2 style="color:#2d3748;">✅ Prediction Complete</h2>
                            <div style="display:flex; gap:16px; justify-content:center; align-items:flex-start; flex-wrap:wrap; margin: 10px 0 6px;">
                                <div>
                                    <p style="margin:6px 0; color:#444; font-weight:600;">Original</p>
                                    <img src="${uploadResult.file_url}" width="300" style="border-radius:12px;">
                                </div>
                                <div>
                                    <p style="margin:6px 0; color:#444; font-weight:600;">Enhanced</p>
                                    <img src="${predictResult.enhanced_url || ''}" width="300" style="border-radius:12px;">
                                </div>
                            </div>
                            <div style="background:#f9fafb; padding:15px; border-radius:12px; margin-top:10px;">
                                <p style="font-size:18px; color:#333;"><b>Disease:</b> ${predictResult.prediction}</p>
                                <p style="font-size:18px; color:#333;"><b>Confidence:</b> ${(predictResult.confidence * 100).toFixed(2)}%</p>
                            </div>
                            <div style="margin-top:10px;">
                                ${predictResult.enhanced_url ? `<a href="${predictResult.enhanced_url}" download style="text-decoration:none; color:#2563eb; font-weight:600;">⬇️ Download enhanced image</a>` : ''}
                            </div>
                            <button id="new-upload-btn" style="
                                margin-top:20px;
                                background:#3b82f6;
                                color:white;
                                border:none;
                                padding:10px 20px;
                                border-radius:8px;
                                cursor:pointer;
                                font-size:16px;
                            ">🔁 Upload Another</button>
                        `;

                        document.querySelector("#new-upload-btn").addEventListener("click", () => {
                            card.remove();
                        });
                    }

                } catch (err) {
                    console.error("Upload Error:", err);
                    alert("❌ Something went wrong!");
                }
            };

            input.click();
        });
    }
});
