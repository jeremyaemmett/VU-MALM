import os
import params

def prepare_output_files(layer_thicknesses, depths):

    data_directory = params.data_directory

    if not os.path.exists(data_directory + 'modelOutput/' + params.site):
        os.mkdir(data_directory + 'modelOutput/' + params.site)

    with open(data_directory+'modelOutput/'+params.site+'/substrate_output.txt', "w") as myfile1:
        myfile1.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile1.close()
    with open(data_directory+'modelOutput/'+params.site+'/microbe_output.txt', "w") as myfile2:
        myfile2.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile2.close()
    with open(data_directory+'modelOutput/'+params.site+'/flux_output.txt', "w") as myfile3:
        myfile3.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile3.close()
    with open(data_directory+'modelOutput/'+params.site+'/envir_output.txt', "w") as myfile4:
        myfile4.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile4.close()
    with open(data_directory+'modelOutput/'+params.site+'/substrate_net_production_output.txt', "w") as myfile5:
        myfile5.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile5.close()
    with open(data_directory+'modelOutput/'+params.site+'/substrate_net_consumption_output.txt', "w") as myfile6:
        myfile6.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile6.close()
    with open(data_directory+'modelOutput/'+params.site+'/substrate_production_output.txt', "w") as myfile7:
        myfile7.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile7.close()
    with open(data_directory+'modelOutput/'+params.site+'/substrate_consumption_output.txt', "w") as myfile8:
        myfile8.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile8.close()
    with open(data_directory+'modelOutput/'+params.site+'/microbe_net_output.txt', "w") as myfile9:
        myfile9.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile9.close()
    with open(data_directory+'modelOutput/'+params.site+'/microbe_growth_output.txt', "w") as myfile10:
        myfile10.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile10.close()
    with open(data_directory+'modelOutput/'+params.site+'/microbe_mortality_output.txt', "w") as myfile11:
        myfile11.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile11.close()
    with open(data_directory+'modelOutput/'+params.site+'/transport_profile_output.txt', "w") as myfile12:
        myfile12.write(f"{len(layer_thicknesses)},{depths[1] - depths[0]}\n")
    myfile12.close()

    return()


def write_output(test_datetime, ca, ga, df, pf, forcing_t, ph_profile, cr, gr, dr, pr):

    data_directory = params.data_directory

    if not os.path.exists(data_directory + 'modelOutput/' + params.site):
        os.mkdir(data_directory + 'modelOutput/' + params.site)

    with open(data_directory+'modelOutput/'+params.site+'/substrate_output.txt', "a") as myfile1:
        myfile1.write(
            f"{test_datetime},{','.join(map(str, ca['ch4']))},{','.join(map(str, ca['co2']))},{','.join(map(str, ca['o2']))},{','.join(map(str, ca['h2']))},{','.join(map(str, ca['ace']))},{','.join(map(str, ca['c']))},{','.join(map(str, ca['doc']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/microbe_output.txt', "a") as myfile2:
        myfile2.write(
            f"{test_datetime},{','.join(map(str, ga['acemethanogen_genes']))},{','.join(map(str, ga['h2_methanogen_genes']))},{','.join(map(str, ga['homoacetogen_genes']))},{','.join(map(str, ga['methanotroph_genes']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/flux_output.txt', "a") as myfile3:
        myfile3.write(str(test_datetime) + ','
                      + str(df['f_diff_ch4']) + ',' + str(df['f_diff_co2']) + ',' + str(df['f_diff_h2']) + ',' + str(df['f_diff_o2']) + ','
                      + str(pf['plant_flux_ch4']) + ',' + str(pf['plant_flux_o2']) + ',' + str(pf['plant_flux_co2']) + ','
                      + str(pf['plant_flux_h2']) + ',' + str(pf['plant_flux_ace']))
        myfile3.write('\n')

    with open(data_directory+'modelOutput/'+params.site+'/envir_output.txt', "a") as myfile4:
        myfile4.write(
            f"{test_datetime},{','.join(map(str, forcing_t['temp_data']))},{','.join(map(str, forcing_t['waterIce_data']))},{','.join(map(str, ph_profile))}\n")
        #print(forcing_t['temp_data'][0])

    with open(data_directory+'modelOutput/'+params.site+'/substrate_net_production_output.txt', "a") as myfile5:
        myfile5.write(
            f"{test_datetime},{','.join(map(str, cr['ch4_c_prod']))},{','.join(map(str, cr['co2_c_prod']))},{','.join(map(str, cr['o2_c_prod']))},{','.join(map(str, cr['h2_c_prod']))},{','.join(map(str, cr['ace_c_prod']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/substrate_net_consumption_output.txt', "a") as myfile6:
        myfile6.write(
            f"{test_datetime},{','.join(map(str, cr['ch4_c_cons']))},{','.join(map(str, cr['co2_c_cons']))},{','.join(map(str, cr['o2_c_cons']))},{','.join(map(str, cr['h2_c_cons']))},{','.join(map(str, cr['ace_c_cons']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/substrate_production_output.txt', "a") as myfile7:
        myfile7.write(
            f"{test_datetime},{','.join(map(str, cr['ch4_c_prod_acmg']))},{','.join(map(str, cr['ch4_c_prod_h2mg']))},{','.join(map(str, cr['ace_c_prod_aesp']))},{','.join(map(str, cr['ace_c_prod_hoag']))},{','.join(map(str, cr['co2_c_prod_ansp']))},{','.join(map(str, cr['co2_c_prod_aesp']))},{','.join(map(str, cr['co2_c_prod_acmg']))},{','.join(map(str, cr['h2_c_prod_ansp']))},{','.join(map(str, cr['o2_c_prod_none']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/substrate_consumption_output.txt', "a") as myfile8:
        myfile8.write(
            f"{test_datetime},{','.join(map(str, cr['ch4_c_cons_mtox']))},{','.join(map(str, cr['ace_c_cons_acmg']))},{','.join(map(str, cr['co2_c_cons_hoag']))},{','.join(map(str, cr['co2_c_cons_h2mg']))},{','.join(map(str, cr['h2_c_cons_hoag']))},{','.join(map(str, cr['h2_c_cons_h2mg']))},{','.join(map(str, cr['o2_c_cons_mtox']))},{','.join(map(str, cr['o2_c_cons_aero']))},{','.join(map(str, cr['o2_c_cons_aesp']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/microbe_net_output.txt', "a") as myfile9:
        myfile9.write(
            f"{test_datetime},{','.join(map(str, gr['acemethanogen_gene_rate']))},{','.join(map(str, gr['h2_methanogen_gene_rate']))},{','.join(map(str, gr['homoacetogen_gene_rate']))},{','.join(map(str, gr['methanotroph_gene_rate']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/microbe_growth_output.txt', "a") as myfile10:
        myfile10.write(
            f"{test_datetime},{','.join(map(str, gr['acemethanogen_gene_prod']))},{','.join(map(str, gr['h2_methanogen_gene_prod']))},{','.join(map(str, gr['homoacetogen_gene_prod']))},{','.join(map(str, gr['methanotroph_gene_prod']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/microbe_mortality_output.txt', "a") as myfile11:
        myfile11.write(
            f"{test_datetime},{','.join(map(str, gr['acemethanogen_gene_mort']))},{','.join(map(str, gr['h2_methanogen_gene_mort']))},{','.join(map(str, gr['homoacetogen_gene_mort']))},{','.join(map(str, gr['methanotroph_gene_mort']))}\n")

    with open(data_directory+'modelOutput/'+params.site+'/transport_profile_output.txt', "a") as myfile12:
        myfile12.write(
            f"{test_datetime},{','.join(map(str, dr['q_diff_ch4_profile']))},{','.join(map(str, dr['q_diff_co2_profile']))},{','.join(map(str, dr['q_diff_o2_profile']))},{','.join(map(str, dr['q_diff_h2_profile']))},{','.join(map(str, dr['q_diff_ace_profile']))},{','.join(map(str, pr['q_plant_ch4_profile']))},{','.join(map(str, pr['q_plant_co2_profile']))},{','.join(map(str, pr['q_plant_o2_profile']))},{','.join(map(str, pr['q_plant_h2_profile']))},{','.join(map(str, pr['q_plant_ace_profile']))}\n")

    return()