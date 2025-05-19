# Conditional Time Series Generation for Streamflow Prediction

This repository contains the project "Conditional time series generation using deep generative model like flow model or diffusive model," submitted in partial fulfillment of the requirements for the Degree of Bachelor of Technology at the Indian Institute of Technology Palakkad.

**Authors:**
*   Sasidhar Reddy Navuluri (122101025)
*   Mantripragda Venakta Karthikeya (112101027)

**Supervisor:**
*   Dr. Sahely Bhadra (Associate Professor, Department of Data Science / Computer Science & Engineering, IIT Palakkad)

## Project Overview

This project focuses on generating conditional time series data, specifically for streamflow (SF) prediction in gauged and ungauged basins. Accurate streamflow forecasting is crucial for water resource management, flood control, drought mitigation, and understanding hydrological impacts of climate change. The project leverages deep generative models, particularly a diffusion-based model (Diffusion-TS), to capture complex temporal dynamics and generate realistic streamflow patterns conditioned on both temporal meteorological data and static non-temporal catchment attributes.

The primary goal is to develop a model capable of:
1.  Predicting streamflow in ungauged basins using non-temporal features (e.g., catchment characteristics) and temporal data from gauged basins.
2.  Integrating temporal and non-temporal data effectively to enhance model performance and accuracy.
3.  Utilizing advanced deep generative models (diffusion-based) for capturing complex streamflow dynamics.
4.  Transferring knowledge from gauged to ungauged basins by quantifying basin similarity based on attributes.

## Dataset

The project utilizes the **CAMELS (Catchment Attributes and MEteorology for Large-sample Studies) dataset**.
*   It contains hydrometeorological data from 671 basins in the US.
*   Daily streamflow records from January 1, 1980, to December 31, 2014.
*   **Temporal Attributes Used:** Precipitation, temperature, solar radiation, daylength, snow water equivalent, vapor pressure.
*   **Non-Temporal (Static) Attributes Used:** Catchment characteristics such as gauge latitude/longitude, mean elevation, slope, area, aridity, soil depth, soil porosity, land cover (forest fraction), geological permeability, etc.

## Methodology

The core methodology revolves around enhancing the **Diffusion-TS** model (Yuan and Qiao, 2023) for conditional time series generation. The model employs an encoder-decoder Transformer architecture.

**Key aspects of the approach:**

1.  **Diffusion Process:**
    *   **Forward Process:** Gradually adds Gaussian noise to the clean streamflow data until it becomes pure noise.
    *   **Reverse Process:** A neural network learns to denoise the noisy data iteratively to reconstruct the original streamflow. This reverse process is conditioned on external information.

2.  **Conditional Generation:**
    *   **Temporal Conditioning:** The model is conditioned on dynamic temporal inputs like precipitation and temperature data.
    *   **Non-Temporal Conditioning:** Static catchment attributes (e.g., soil type, elevation) are integrated as conditioning factors. These attributes are embedded and fused with the encoded temporal information to guide the decoder in generating basin-specific streamflow.

3.  **Model Architecture (Transformer-based):**
    *   **Encoder:** Processes temporal conditioning data (e.g., meteorological variables) to create a latent representation.
    *   **Decoder:** Takes the noisy streamflow data, the encoded temporal representation, and processed non-temporal basin attributes as input to predict the denoised streamflow.
    *   The architecture was modified to include a pathway for non-temporal data, which involves expanding these features across the sequence length, embedding them, and merging them with encoded temporal information.
    *   An explicit seasonality block was removed from the original Diffusion-TS decoder to encourage learning complex temporal relationships directly from the data and conditioning variables.

4.  **Training and Evaluation:**
    *   The model was trained on data from 533 different gauge basins.
    *   Evaluation was performed on 30 unseen gauge basins to assess generalization capabilities.

## Implementation Evolution

*   **Previous Work:**
    *   Initial implementation focused on a smaller dataset (single gauge or few gauges).
    *   The dataset was expanded to 533 gauge points for training.
    *   The seasonality block in the decoder was removed to simplify the model and improve focus on learned temporal relationships.
    *   Only temporal data was used for conditioning.

*   **Current Work:**
    *   **Integration of Non-Temporal Data:** Static basin features were incorporated as conditioning factors alongside dynamic temporal data. This allows the model to generate streamflow patterns that are more physically accurate and basin-specific.
    *   **Enhanced Conditional Generation:** The diffusion model now produces streamflow conditioned on both time-varying inputs (e.g., temperature) and the distinct, static characteristics of each individual basin.
    *   **Non-Temporal Data Pathway:** The Transformer architecture was modified with specialized layers to process non-temporal input, expand it across the sequence length, embed it, and merge it with encoded temporal information to condition the decoder.

## Results

The model demonstrated strong generalization by effectively generating basin-specific streamflow sequences for 30 previously unexplored basins.
*   The integration of non-temporal data led to improved performance in capturing basin-specific characteristics.
*   Example RMSE for test gauges (with non-temporal data):
    *   Gauge 29: RMSE 0.0677
    *   Gauge 30: RMSE 0.1810
*   Qualitative results show that the generated streamflow aligns well with observed patterns.

(Visualizations of original vs. reconstructed streamflow plots would typically be linked or embedded here if available as image files.)

## Conclusion and Future Work

The developed conditional diffusion-based model (Diffusion-TS) shows significant promise for streamflow prediction, especially in ungauged or data-scarce scenarios. Key improvements include training on a larger multi-basin dataset and effectively combining static basin attributes with dynamic temporal variables.



## References

*   Yuan, X., and Qiao, Y. (2023). "Diffusion-ts: Interpretable diffusion for general time series generation." Hefei University of Technology.
*   Bhattacharjee, S. (2023). "Streamflow prediction in ungauged basins." (Likely a related or foundational internal project/report).
*   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is all you need." In *Advances in Neural Information Processing Systems* (NeurIPS).
