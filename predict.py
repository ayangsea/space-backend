import lightkurve as lk
from lightkurve.periodogram import LombScarglePeriodogram
from openai import OpenAI

def predict(name):
    # Search for a light curve (replace 'TIC 25155310' with your target)
    search_result = lk.search_lightcurve(name, mission='TESS')

    lc = search_result.download()
    print(type(lc))
    lc_clean = lc.flatten(window_length=401).remove_outliers()
    pg = lc_clean.to_periodogram(method="lombscargle", oversample_factor=10)
    period = pg.period_at_max_power.value
    amplitude = lc_clean.flux.max() - lc_clean.flux.min()
    mean_flux = lc_clean.flux.mean()
    variability = lc_clean.flux.std()

    features = {
        "period": period,
        "amplitude": float(amplitude),
        "mean_flux": float(mean_flux),
        "variability": float(variability)
    }

    prompt = f"""
    Analyze the following light curve data and classify the type of variable star. Provide a one-word answer only.

    Features:
    - Period: {features['period']:.3f} days
    - Amplitude: {features['amplitude']:.3f} normalized flux
    - Mean Flux: {features['mean_flux']:.3f}
    - Variability: {features['variability']:.3f}

    Answer with just one word, the type of star.
    """

    # Set your OpenAI API key
    key = ""

    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = completion.choices[0].message.content

    return answer