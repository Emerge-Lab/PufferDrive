#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pufferlib/ocean/drive/drive.h"

#define OUT_DIR "collision_responsibility_viz"
#define TARGET_IDX 0
#define ADV_IDX 1

typedef enum ExpectedResponsibility {
    EXPECT_TARGET_LOW,
    EXPECT_TARGET_HIGH,
    EXPECT_TARGET_SPLIT,
} ExpectedResponsibility;

typedef struct Scenario {
    const char *name;
    const char *description;
    Agent target;
    Agent adv;
    ExpectedResponsibility expected;
    float low_threshold;
    float high_threshold;
} Scenario;

typedef struct ScenarioResult {
    int collided;
    float normal_x;
    float normal_y;
    float sat_normal_x;
    float sat_normal_y;
    float penetration;
    float target_responsibility;
    float adv_responsibility;
    float target_dot;
    float adv_dot;
    int target_impact_zone;
    int adv_impact_zone;
    int passed;
} ScenarioResult;

static void set_pose(Agent *agent, float x, float y, float heading, float speed) {
    agent->sim_x = x;
    agent->sim_y = y;
    agent->sim_z = 0.0f;
    agent->sim_heading = heading;
    agent->cos_heading = cosf(heading);
    agent->sin_heading = sinf(heading);
    agent->sim_vx = speed * agent->cos_heading;
    agent->sim_vy = speed * agent->sin_heading;
    agent->sim_speed = fabsf(speed);
    agent->sim_speed_signed = speed;
    agent->sim_length = 4.6f;
    agent->sim_width = 2.0f;
    agent->sim_height = 1.6f;
    agent->sim_valid = 1;
    agent->type = VEHICLE;
}

static void set_prev(Agent *agent, float x, float y, float heading) {
    agent->prev_sim_x = x;
    agent->prev_sim_y = y;
    agent->prev_sim_heading = heading;
}

static Scenario make_scenario(const char *name, const char *description, Agent target, Agent adv,
                              ExpectedResponsibility expected, float low_threshold, float high_threshold) {
    Scenario scenario;
    memset(&scenario, 0, sizeof(scenario));
    scenario.name = name;
    scenario.description = description;
    scenario.target = target;
    scenario.adv = adv;
    scenario.expected = expected;
    scenario.low_threshold = low_threshold;
    scenario.high_threshold = high_threshold;
    return scenario;
}

static Scenario make_rear_end(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 2.0f, 0.0f, 0.0f, 8.0f);
    set_prev(&target, 1.2f, 0.0f, 0.0f);
    set_pose(&adv, -1.6f, 0.0f, 0.0f, 16.0f);
    set_prev(&adv, -3.2f, 0.0f, 0.0f);
    return make_scenario("rear_end_faster_adversary", "Adversary closes from behind into target rear.", target, adv,
                         EXPECT_TARGET_LOW, 0.25f, 0.75f);
}

static Scenario make_stopped_target_hit(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 0.0f, 0.0f, 0.0f, 0.0f);
    set_prev(&target, 0.0f, 0.0f, 0.0f);
    set_pose(&adv, -3.5f, 0.0f, 0.0f, 15.0f);
    set_prev(&adv, -5.0f, 0.0f, 0.0f);
    return make_scenario("stopped_target_hit_from_behind", "Stopped target is hit by moving adversary.", target, adv,
                         EXPECT_TARGET_LOW, 0.10f, 0.90f);
}

static Scenario make_target_front_impact(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 2.7f, 0.0f, 0.0f, 15.0f);
    set_prev(&target, 1.2f, 0.0f, 0.0f);
    set_pose(&adv, 6.2f, 0.0f, 0.0f, 0.0f);
    set_prev(&adv, 6.2f, 0.0f, 0.0f);
    return make_scenario("target_front_hits_stopped_vehicle", "Target drives into a stopped vehicle ahead.", target,
                         adv, EXPECT_TARGET_HIGH, 0.25f, 0.75f);
}

static Scenario make_clean_t_bone(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 0.0f, 0.0f, 0.0f, 8.0f);
    set_prev(&target, -0.8f, 0.0f, 0.0f);
    set_pose(&adv, 1.2f, 1.8f, -M_PI_2, 12.0f);
    set_prev(&adv, 1.2f, 3.0f, -M_PI_2);
    return make_scenario("clean_t_bone_adversary_into_side", "Adversary moves laterally into target side.", target, adv,
                         EXPECT_TARGET_LOW, 0.25f, 0.75f);
}

static Scenario make_cut_in_target_hits_intruder(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 1.5f, 0.0f, 0.0f, 15.0f);
    set_prev(&target, 0.0f, 0.0f, 0.0f);
    set_pose(&adv, 3.3f, 0.85f, -0.35f, 8.0f);
    set_prev(&adv, 2.55f, 1.12f, -0.35f);
    return make_scenario("cut_in_intruder_ahead_target_hits", "Slower adversary cuts in ahead; target hits intruder.",
                         target, adv, EXPECT_TARGET_HIGH, 0.25f, 0.75f);
}

static Scenario make_cut_in_adversary_hits_target_rear_side(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 1.5f, 0.0f, 0.0f, 15.0f);
    set_prev(&target, 0.0f, 0.0f, 0.0f);
    set_pose(&adv, 0.1f, 0.95f, -0.72f, 8.0f);
    set_prev(&adv, -0.05f, 1.58f, -0.72f);
    return make_scenario("cut_in_adversary_rear_side_graze", "Adversary cuts into target rear-side/corner.", target,
                         adv, EXPECT_TARGET_LOW, 0.25f, 0.75f);
}

static Scenario make_cut_in_center_bias_probe(void) {
    Agent target = {0};
    Agent adv = {0};
    set_pose(&target, 1.5f, 0.0f, 0.0f, 15.0f);
    set_prev(&target, 0.0f, 0.0f, 0.0f);
    set_pose(&adv, 2.15f, 0.98f, -0.95f, 8.0f);
    set_prev(&adv, 2.61f, 1.63f, -0.95f);
    return make_scenario("cut_in_center_ahead_side_graze_probe",
                         "Adversary center is ahead, but motion is a side/corner cut-in.", target, adv,
                         EXPECT_TARGET_LOW, 0.25f, 0.75f);
}

static int expectation_passed(const Scenario *scenario, float target_rho) {
    if (scenario->expected == EXPECT_TARGET_LOW)
        return target_rho <= scenario->low_threshold;
    if (scenario->expected == EXPECT_TARGET_HIGH)
        return target_rho >= scenario->high_threshold;
    return target_rho > scenario->low_threshold && target_rho < scenario->high_threshold;
}

static ScenarioResult evaluate_scenario(const Scenario *scenario) {
    ScenarioResult result;
    memset(&result, 0, sizeof(result));
    Agent target = scenario->target;
    Agent adv = scenario->adv;

    result.collided =
        check_obb_collision_with_normal(&target, &adv, &result.sat_normal_x, &result.sat_normal_y, &result.penetration);
    if (!result.collided) {
        result.passed = 0;
        return result;
    }

    compute_collision_normal(&target, &adv, result.sat_normal_x, result.sat_normal_y, &result.normal_x,
                             &result.normal_y);
    evaluate_collision_pair(&target, &adv, result.normal_x, result.normal_y, NULL, &result.target_responsibility);
    result.adv_responsibility = 1.0f - result.target_responsibility;
    result.target_dot = target.sim_vx * result.normal_x + target.sim_vy * result.normal_y;
    result.adv_dot = adv.sim_vx * result.normal_x + adv.sim_vy * result.normal_y;
    result.target_impact_zone = classify_impact_zone_from_normal(&target, result.normal_x, result.normal_y);
    result.adv_impact_zone = classify_impact_zone_from_normal(&adv, -result.normal_x, -result.normal_y);
    result.passed = expectation_passed(scenario, result.target_responsibility);
    return result;
}

static void write_poly(FILE *file, const Agent *agent, int previous, const char *stroke, const char *fill, float scale,
                       float origin_x, float origin_y) {
    Agent copy = *agent;
    if (previous) {
        copy.sim_x = agent->prev_sim_x;
        copy.sim_y = agent->prev_sim_y;
        copy.sim_heading = agent->prev_sim_heading;
        copy.cos_heading = cosf(copy.sim_heading);
        copy.sin_heading = sinf(copy.sim_heading);
    }

    float corners[4][2];
    compute_agent_corners(&copy, corners);
    fprintf(file, "<polygon points=\"");
    for (int i = 0; i < 4; i++) {
        float sx = origin_x + scale * corners[i][0];
        float sy = origin_y - scale * corners[i][1];
        fprintf(file, "%0.2f,%0.2f ", sx, sy);
    }
    fprintf(file, "\" fill=\"%s\" stroke=\"%s\" stroke-width=\"2\" />\n", fill, stroke);
}

static void write_line(FILE *file, float x1, float y1, float x2, float y2, const char *color, float width, float scale,
                       float origin_x, float origin_y) {
    fprintf(file,
            "<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" stroke=\"%s\" stroke-width=\"%0.2f\" "
            "marker-end=\"url(#arrow)\" />\n",
            origin_x + scale * x1, origin_y - scale * y1, origin_x + scale * x2, origin_y - scale * y2, color, width);
}

static void write_scenario_svg(const Scenario *scenario, const ScenarioResult *result) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.svg", OUT_DIR, scenario->name);
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        perror(path);
        return;
    }

    const float scale = 55.0f;
    const float origin_x = 330.0f;
    const float origin_y = 260.0f;
    fprintf(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"520\" viewBox=\"0 0 720 520\">\n");
    fprintf(file, "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"4\" "
                  "orient=\"auto\"><path d=\"M0,0 L8,4 L0,8 z\" fill=\"context-stroke\" /></marker></defs>\n");
    fprintf(file, "<rect width=\"720\" height=\"520\" fill=\"#fbfbfb\" />\n");
    fprintf(file, "<line x1=\"0\" y1=\"%0.2f\" x2=\"720\" y2=\"%0.2f\" stroke=\"#dddddd\" />\n", origin_y, origin_y);

    write_poly(file, &scenario->target, 1, "#2f6fed", "rgba(47,111,237,0.10)", scale, origin_x, origin_y);
    write_poly(file, &scenario->adv, 1, "#d66b00", "rgba(214,107,0,0.10)", scale, origin_x, origin_y);
    write_poly(file, &scenario->target, 0, "#0f47b5", "rgba(47,111,237,0.28)", scale, origin_x, origin_y);
    write_poly(file, &scenario->adv, 0, "#a34e00", "rgba(214,107,0,0.28)", scale, origin_x, origin_y);

    write_line(file, scenario->target.sim_x, scenario->target.sim_y,
               scenario->target.sim_x + 0.12f * scenario->target.sim_vx,
               scenario->target.sim_y + 0.12f * scenario->target.sim_vy, "#0f47b5", 3.0f, scale, origin_x, origin_y);
    write_line(file, scenario->adv.sim_x, scenario->adv.sim_y, scenario->adv.sim_x + 0.12f * scenario->adv.sim_vx,
               scenario->adv.sim_y + 0.12f * scenario->adv.sim_vy, "#a34e00", 3.0f, scale, origin_x, origin_y);
    write_line(file, scenario->target.sim_x, scenario->target.sim_y, scenario->target.sim_x + 1.5f * result->normal_x,
               scenario->target.sim_y + 1.5f * result->normal_y, "#cc0000", 4.0f, scale, origin_x, origin_y);
    write_line(file, scenario->target.sim_x, scenario->target.sim_y,
               scenario->target.sim_x + 1.2f * result->sat_normal_x,
               scenario->target.sim_y + 1.2f * result->sat_normal_y, "#7b2cbf", 2.0f, scale, origin_x, origin_y);

    fprintf(file, "<text x=\"24\" y=\"32\" font-family=\"monospace\" font-size=\"18\">%s</text>\n", scenario->name);
    fprintf(file, "<text x=\"24\" y=\"58\" font-family=\"monospace\" font-size=\"13\">%s</text>\n",
            scenario->description);
    fprintf(file,
            "<text x=\"24\" y=\"84\" font-family=\"monospace\" font-size=\"13\">target rho=%0.3f, adv rho=%0.3f, "
            "%s</text>\n",
            result->target_responsibility, result->adv_responsibility, result->passed ? "PASS" : "MISMATCH");
    fprintf(file,
            "<text x=\"24\" y=\"106\" font-family=\"monospace\" font-size=\"13\">normal=(%0.2f,%0.2f), "
            "sat=(%0.2f,%0.2f), dots target/adv=(%0.2f,%0.2f)</text>\n",
            result->normal_x, result->normal_y, result->sat_normal_x, result->sat_normal_y, result->target_dot,
            result->adv_dot);
    fprintf(file, "<text x=\"24\" y=\"490\" font-family=\"monospace\" font-size=\"12\" fill=\"#555\">blue=target, "
                  "orange=adversary, pale=previous, filled=current, red=selected normal, purple=SAT normal</text>\n");
    fprintf(file, "</svg>\n");
    fclose(file);
}

int main(void) {
    Scenario scenarios[] = {
        make_rear_end(),
        make_stopped_target_hit(),
        make_target_front_impact(),
        make_clean_t_bone(),
        make_cut_in_target_hits_intruder(),
        make_cut_in_adversary_hits_target_rear_side(),
        make_cut_in_center_bias_probe(),
    };
    const int num_scenarios = (int)(sizeof(scenarios) / sizeof(scenarios[0]));

    if (system("mkdir -p " OUT_DIR) != 0) {
        fprintf(stderr, "Failed to create %s\n", OUT_DIR);
        return 1;
    }

    int failures = 0;
    printf("Collision responsibility normal sanity cases\n");
    printf("SVG outputs: tests/drive/%s/*.svg\n\n", OUT_DIR);

    for (int i = 0; i < num_scenarios; i++) {
        ScenarioResult result = evaluate_scenario(&scenarios[i]);
        write_scenario_svg(&scenarios[i], &result);
        if (!result.passed)
            failures++;

        printf("%-40s %s target_rho=%0.3f adv_rho=%0.3f n=(%0.2f,%0.2f) sat=(%0.2f,%0.2f)\n", scenarios[i].name,
               result.passed ? "PASS    " : "MISMATCH", result.target_responsibility, result.adv_responsibility,
               result.normal_x, result.normal_y, result.sat_normal_x, result.sat_normal_y);
    }

    if (failures > 0) {
        printf("\n%d scenario(s) disagree with the expected responsibility direction.\n", failures);
        return 1;
    }

    printf("\nAll responsibility sanity cases matched expectations.\n");
    return 0;
}
