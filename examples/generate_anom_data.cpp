#include <iostream>                                                                                                                                                                                   
#include <vector>
#include <TRandom3.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>

using namespace std;

vector<double> generateNormalData(int numEvents) {
    TRandom3 randGen;
    vector<double> data;
    for (int i = 0; i < numEvents; ++i) {
        // Simulate normal response (i.e. Gaussian distribution)
        double response = randGen.Gaus(0, 1);
        data.push_back(response);
    }
    return data;
}

// Inject anomalies into normal data
void injectAnomalies(vector<double>& data, int numAnomalies) {
    TRandom3 randGen;
    int dataSize = data.size();
    for (int i = 0; i < numAnomalies; ++i) {
        int index = randGen.Integer(dataSize);
        data[index] = randGen.Uniform(5, 10);
    }
}

int main() {
    int numNormalEvents = 100000;
    int numAnomalies = 2000;

    vector<double> normalData = generateNormalData(numNormalEvents);

    injectAnomalies(normalData, numAnomalies);

    TFile outputFile("mixedData.root", "RECREATE");
    TTree tree("tree", "Particle Detector Responses");

    double response;
    tree.Branch("response", &response);

    for (const auto& dataPoint : normalData) {
        response = dataPoint;
        tree.Fill();
    }

    tree.Write();
    outputFile.Close();

    TH1D histAfterInjection("response_histogram_after_injection", "Response Distribution (After Injection)", 100, -10, 10);
    for (const auto& dataPoint : normalData) {
        histAfterInjection.Fill(dataPoint);
    }

    TCanvas canvasAfter("canvas_after_injection", "Response Histogram (After Injection)", 800, 600);
    histAfterInjection.Draw();
    histAfterInjection.GetXaxis()->SetTitle("Response");
    histAfterInjection.GetYaxis()->SetTitle("Frequency");
    canvasAfter.Draw();
    canvasAfter.SaveAs("response_histogram_after_injection.png");

    return 0;
}
