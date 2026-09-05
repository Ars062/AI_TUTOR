function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for (var i in uiBathrooms) {
        if (uiBathrooms[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1;
}

function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for (var i in uiBHK) {
        if (uiBHK[i].checked) {
            return parseInt(i) + 1;
        }
    }
    return -1;
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5001/api/predict_home_price";
    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value
    }, function (data, status) {
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
        console.log(status);
    });
}

function onClickedGetRiskReport() {
    console.log("Risk report button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var price = document.getElementById("uiPrice");
    var riskReport = document.getElementById("riskReport");

    if (!location.value) {
        alert("Please select a location first");
        return;
    }

    var url = "http://127.0.0.1:5001/api/get_risk_report";
    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value,
        price: parseFloat(price.value)
    }, function (data, status) {
        console.log("Risk report received:", data);
        displayRiskReport(data);
        riskReport.classList.remove("hidden");
        riskReport.scrollIntoView({ behavior: 'smooth' });
    }).fail(function(xhr, status, error) {
        console.error("Error getting risk report:", error);
        alert("Error getting risk report. Please try again.");
    });
}

function displayRiskReport(data) {
    var risk = data.investment_risk;
    var overpricing = data.overpricing_analysis;
    var locationRating = data.location_rating;
    var roi = data.roi_estimate;

    var scoreEl = document.getElementById("riskScore");
    scoreEl.innerHTML = '<span class="score-value">' + risk.score + '</span><span class="score-label"> / 10</span>';
    scoreEl.className = "risk-gauge score-" + getScoreClass(risk.score);

    document.getElementById("riskLevel").textContent = risk.level;
    document.getElementById("riskLevel").className = "risk-level level-" + getScoreClass(risk.score);

    var reasonsHtml = "";
    if (risk.reasons && risk.reasons.length > 0) {
        reasonsHtml = "<ul>";
        risk.reasons.forEach(function(reason) {
            reasonsHtml += "<li>" + reason + "</li>";
        });
        reasonsHtml += "</ul>";
    }
    document.getElementById("riskReasons").innerHTML = reasonsHtml;

    document.getElementById("overpricingStatus").textContent = overpricing.status;
    document.getElementById("overpricingStatus").className = "status-badge status-" + getStatusClass(overpricing.status);
    document.getElementById("overpricingDeviation").textContent = overpricing.deviation;
    document.getElementById("overpricingMessage").textContent = overpricing.message;
    document.getElementById("yourRate").textContent = "₹" + overpricing.your_rate + "/sqft";
    document.getElementById("marketRate").textContent = "₹" + overpricing.market_rate + "/sqft";

    document.getElementById("locationGrade").textContent = locationRating.grade;
    document.getElementById("locationGrade").className = "grade-badge grade-" + locationRating.grade;
    document.getElementById("locationDesc").textContent = locationRating.description;

    if (locationRating.breakdown) {
        var b = locationRating.breakdown;
        if (b.safety_index) document.getElementById("safetyScore").textContent = b.safety_index.score + "/10 (" + b.safety_index.rating + ")";
        if (b.flood_risk) document.getElementById("floodRisk").textContent = b.flood_risk.level + " - " + b.flood_risk.impact;
        if (b.infrastructure_score) document.getElementById("infraScore").textContent = b.infrastructure_score.score + "/10 (" + b.infrastructure_score.rating + ")";
        if (b.market_liquidity) document.getElementById("liquidityLevel").textContent = b.market_liquidity.level + " - " + b.market_liquidity.impact;
        if (b.crime_rate) document.getElementById("crimeLevel").textContent = b.crime_rate.level + " - " + b.crime_rate.impact;
    }

    document.getElementById("monthlyRent").textContent = "₹" + roi.monthly_rent.toLocaleString();
    document.getElementById("rentalYield").textContent = roi.rental_yield;
    document.getElementById("appreciation").textContent = roi.expected_appreciation;
    document.getElementById("totalRoi").textContent = roi.total_roi;
    document.getElementById("paybackYears").textContent = roi.payback_years;

    // Phase 4: P×I Risk Matrix
    if (data.risk_matrix) {
        displayRiskMatrix(data.risk_matrix);
    }
}

function getScoreClass(score) {
    if (score <= 3) return "low";
    if (score <= 5) return "medium";
    if (score <= 7) return "high";
    return "veryhigh";
}

function getStatusClass(status) {
    if (status === "Fair") return "fair";
    if (status === "Underpriced") return "under";
    if (status === "Slightly Low") return "under";
    return "over";
}

function displayRiskMatrix(matrix) {
    // Overall scores
    document.getElementById("piOverallProb").textContent = matrix.overall_probability.toFixed(2);
    document.getElementById("piOverallImpact").textContent = matrix.overall_impact.toFixed(2);
    document.getElementById("piOverallScore").textContent = matrix.overall_risk_score.toFixed(1);

    var levelEl = document.getElementById("piOverallLevel");
    levelEl.textContent = matrix.overall_risk_level;
    levelEl.className = "pi-level-badge pi-level-" + matrix.overall_risk_level.toLowerCase();

    // Category rows
    var rowsHtml = "";
    var categories = ["Financial", "Market", "Location", "Environmental", "Liquidity", "Rental/Income"];

    categories.forEach(function(cat) {
        var a = null;
        matrix.assessments.forEach(function(assess) {
            if (assess.risk_category === cat) a = assess;
        });
        if (!a) return;

        var scoreClass = "pi-score-" + a.risk_level.toLowerCase();
        rowsHtml += '<div class="pi-row">' +
            '<div class="pi-cell pi-cat">' + a.risk_category + '</div>' +
            '<div class="pi-cell pi-num">' + a.probability + '</div>' +
            '<div class="pi-cell pi-desc">' + a.probability_label + '</div>' +
            '<div class="pi-cell pi-num">' + a.impact + '</div>' +
            '<div class="pi-cell pi-desc">' + a.impact_label + '</div>' +
            '<div class="pi-cell pi-num">' + a.risk_score + '</div>' +
            '<div class="pi-cell pi-level"><span class="pi-level-text ' + scoreClass + '">' + a.risk_level + '</span></div>' +
            '</div>';
    });

    document.getElementById("piRows").innerHTML = rowsHtml;
}

function onPageLoad() {
    console.log("document loaded");
    var url = "http://127.0.0.1:5001/api/get_location_names";
    $.get(url, function (data, status) {
        console.log("got response for get_location_names request");
        if (data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            var defaultOpt = new Option("Choose a Location", "");
            defaultOpt.disabled = true;
            defaultOpt.selected = true;
            $('#uiLocations').append(defaultOpt);
            for (var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });
}

window.onload = onPageLoad;
