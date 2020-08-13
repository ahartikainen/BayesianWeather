
functions {
    real bucket_area(real r, real d_i, real d_o, real d_l) {
        real circle_area = pi() * square(r);
        
        real p = (r * 2 + d_i) / 2;
        real segment_angle = 2 * pi() * d_i / (2 * pi() * r);
        real segment_area = 0.5 * square(r) * (segment_angle - sin(segment_angle));
        
        real wedge_area = d_i * d_l + fabs(d_i - d_o) * d_l;
        real area = circle_area - segment_area + wedge_area;
        return area;
    }
}
data {
    int<lower=0> K;             // # of buckets
    
    // Weigths
    
    int<lower=0> N_b;           // # of observations for weight (bucket)
    vector[N_b] w_b;            // bucket weight observations (bucket) [g]
    int s_b[K];                 // bucket weight group sizes (bucket)

    int<lower=0> N_bw;          // # of observations for weight (bucket + water)
    vector[N_bw] w_bw;          // bucket weight observations (bucket + water) [g]
    int s_bw[K];                // bucket weight group sizes (bucket + water)
    
    // Lenghts
    
    int<lower=0> N_d;           // # of observations for diameter
    vector[N_d] d;              // bucket diameter observations [cm]
    int s_d[K];                 // bucket diameter group sizes
    
    int<lower=0> N_di;          // # of observations for inner length
    vector[N_di] d_i;           // bucket inner length observations [cm]
    int s_di[K];                // bucket inner length group sizes
    
    int<lower=0> N_do;          // # of observations for outer length
    vector[N_do] d_o;           // bucket outer length observations [cm]
    int s_do[K];                // bucket inner length group sizes
    
    int<lower=0> N_dl;          // # of observations for length
    vector[N_dl] d_l;           // bucket length observations [cm]
    int s_dl[K];                // bucket length group sizes
    
}
transformed data {
    real scale_sd = sqrt(2);            // estimated absolute error
    real tape_measure_sd = sqrt(2);     // estimated absolute error
    real density_water = 999;           // kg/m^3 for water at 15 degC 
                                        // real temperature between 
                                        // 15 degC (999 kg/m^3) and 25 degC (997 kg/m^3))
}
parameters {   
    // estimated true values
    vector<lower=0>[K] weight_bucket;
    vector<lower=0>[K] weight_bucket_water;
    vector<lower=0>[K] diameter;
    vector<lower=0>[K] d_inner;
    vector<lower=0>[K] d_outer;
    vector<lower=0>[K] d_length;

    vector<lower=-0.5, upper=0.5>[N_b] w_b_err;
    vector<lower=-0.5, upper=0.5>[N_bw] w_bw_err;
}
transformed parameters {
    vector[N_b] w_b_continuous = w_b + w_b_err;
    vector[N_bw] w_bw_continuous = w_bw + w_bw_err;
    
    vector<lower=0>[K] radius = diameter ./ 2;
    vector<lower=0>[K] weight_water = weight_bucket_water - weight_bucket;
    vector<lower=0>[K] area;
    for (k in 1:K) {
        area[k] = bucket_area(radius[k], d_inner[k], d_outer[k], d_length[k]);
    }
}
model {
    int pos_b = 1;
    int pos_bw = 1;
    int pos_d = 1;
    int pos_di = 1;
    int pos_do = 1;
    int pos_dl = 1;
    
    for (k in 1:K) {
        segment(w_b_continuous, pos_b, s_b[k]) ~ normal(weight_bucket[k], scale_sd);
        pos_b = pos_b + s_b[k];
        
        segment(w_bw_continuous, pos_bw, s_bw[k]) ~ normal(weight_bucket_water[k], scale_sd);
        pos_bw = pos_bw + s_bw[k];
        
        segment(d, pos_d, s_d[k]) ~ normal(diameter[k], tape_measure_sd);
        pos_d = pos_d + s_d[k];
        
        segment(d_i, pos_di, s_di[k]) ~ normal(d_inner[k], tape_measure_sd);
        pos_di = pos_di + s_di[k];
        
        segment(d_o, pos_do, s_do[k]) ~ normal(d_outer[k], tape_measure_sd);
        pos_do = pos_do + s_do[k];
        
        segment(d_l, pos_dl, s_dl[k]) ~ normal(d_length[k], tape_measure_sd);
        pos_dl = pos_dl + s_dl[k];
    }
}
generated quantities {
    // posterior predictive for weights
    vector[N_b] pred_w_b_continuous; // bucket weight observations (bucket) [g]
    vector[N_bw] pred_w_bw_continuous;// bucket weight observations (bucket + water) [g]
    // posterior predictive for lengths
    vector[N_d] pred_d;              // bucket diameter observations [cm]
    vector[N_di] pred_d_i;           // bucket inner length observations [cm]
    vector[N_do] pred_d_o;           // bucket outer length observations [cm]
    vector[N_dl] pred_d_l;           // bucket length observations [cm]
    
    // log-likelihood for weights
    vector[N_b] loglik_w_b_continuous;// bucket weight observations (bucket)
    vector[N_bw] loglik_w_bw_continuous;// bucket weight observations (bucket + water)
    // log-likelihood for lengths
    vector[N_d] loglik_d;              // bucket diameter observations
    vector[N_di] loglik_d_i;           // bucket inner length observations
    vector[N_do] loglik_d_o;           // bucket outer length observations
    vector[N_dl] loglik_d_l;           // bucket length observations
    
    // estimate precipitation [mm]
    vector[K] precipitation;
    
    {
    int pos_b = 1;
    int pos_bw = 1;
    int pos_d = 1;
    int pos_di = 1;
    int pos_do = 1;
    int pos_dl = 1;
    
    for (k in 1:K) {
        // results in mm of precipitation
        // see https://en.wikipedia.org/wiki/Precipitation#Measurement
        precipitation[k] = (weight_water[k] / density_water) / area[k] * 10000;
        
        
        ///////////////////////
        // posterior predictive
        // log-likelihood
        ///////////////////////
        for (n in pos_b:pos_b+s_b[k]-1){
            pred_w_b_continuous[n] = normal_rng(weight_bucket[k], scale_sd);
            loglik_w_b_continuous[n] = normal_lpdf(w_b_continuous[n] | weight_bucket[k], scale_sd);}
        pos_b = pos_b + s_b[k];
        
        for (n in pos_bw:pos_bw+s_bw[k]-1){
            pred_w_bw_continuous[n] = normal_rng(weight_bucket_water[k], scale_sd);
            loglik_w_bw_continuous[n] = normal_lpdf(w_bw_continuous[n] | weight_bucket_water[k], scale_sd);}
        pos_bw = pos_bw + s_bw[k];
        
        for (n in pos_d:pos_d+s_d[k]-1){
            pred_d[n] = normal_rng(diameter[k], tape_measure_sd);
            loglik_d[n] = normal_lpdf(d[n] | diameter[k], tape_measure_sd);}
        pos_d = pos_d + s_d[k];
        
        for (n in pos_di:pos_di+s_di[k]-1){
            pred_d_i[n] = normal_rng(d_inner[k], tape_measure_sd);
            loglik_d_i[n] = normal_lpdf(d_i[n] | d_inner[k], tape_measure_sd);}
        pos_di = pos_di + s_di[k];
        
        for (n in pos_do:pos_do+s_do[k]-1){
            pred_d_o[n] = normal_rng(d_outer[k], tape_measure_sd);
            loglik_d_o[n] = normal_lpdf(d_o[n] | d_outer[k], tape_measure_sd);}
        pos_do = pos_do + s_do[k];
        
        for (n in pos_dl:pos_dl+s_dl[k]-1){
            pred_d_l[n] = normal_rng(d_length[k], tape_measure_sd);
            loglik_d_l[n] = normal_lpdf(d_l[n] | d_length[k], tape_measure_sd);}
        pos_dl = pos_dl + s_dl[k];
    }
    }
}

