karabo_input = KaraboInput()

@Variable function xgm(data -> karabo"SA2_XTD1_XGM/XGM/DOOCS:output[data.intensityTD]")
    data
end
