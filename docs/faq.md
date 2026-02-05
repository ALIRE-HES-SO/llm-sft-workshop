---
icon: lucide/circle-question-mark
---

Frequently Asked Questions

### How can I add an Exoscale instance?

#### Landing page
Log in to your [Exoscale](https://www.exoscale.com) account to access the landing page shown below.

![landing_page](./images/extra/configure_exoscale/landing_page_light.png#only-light)
![landing_page](./images/extra/configure_exoscale/landing_page_dark.png#only-dark)

#### Security Groups setup

![title](./images/extra/configure_exoscale/security_groups_light.png#only-light)

![title](./images/extra/configure_exoscale/security_groups_dark.png#only-dark)

![title](./images/extra/configure_exoscale/title.security_groups_add_light#only-light)
![title](./images/extra/configure_exoscale/security_groups_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/security_groups_name_light.png#only-light)
![title](./images/extra/configure_exoscale/security_groups_name_dark.png#only-dark)

![title](./images/extra/configure_exoscale/security_groups_dots_light.png#only-light)
![title](./images/extra/configure_exoscale/security_groups_dots_dark.png#only-dark)

![title](./images/extra/configure_exoscale/security_groups_details_light.png#only-light)
![title](./images/extra/configure_exoscale/security_groups_details_dark.png#only-dark)

![title](./images/extra/configure_exoscale/security_groups_add_rule_light.png#only-light)
![title](./images/extra/configure_exoscale/security_groups_add_rule_dark.png#only-dark)

![title](./images/extra/configure_exoscale/security_groups_ssh_light.png#only-light)
![title](./images/extra/configure_exoscale/security_groups_ssh_dark.png#only-dark)

#### Private Networks setup

![title](./images/extra/configure_exoscale/private_networks_light.png#only-light)
![title](./images/extra/configure_exoscale/private_networks_dark.png#only-dark)

![title](./images/extra/configure_exoscale/private_networks_add_light.png#only-light)
![title](./images/extra/configure_exoscale/private_networks_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/private_networks_name_light.png#only-light)
![title](./images/extra/configure_exoscale/private_networks_name_dark.png#only-dark)

![title](./images/extra/configure_exoscale/private_networks_name_add_light.png#only-light)
![title](./images/extra/configure_exoscale/private_networks_name_add_dark.png#only-dark)

#### SSH Keys setup

![title](./images/extra/configure_exoscale/ssh_keys_light.png#only-light)
![title](./images/extra/configure_exoscale/ssh_keys_dark.png#only-dark)

![title](./images/extra/configure_exoscale/ssh_keys_add_light.png#only-light)
![title](./images/extra/configure_exoscale/ssh_keys_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/ssh_keys_import_light.png#only-light)
![title](./images/extra/configure_exoscale/ssh_keys_import_dark.png#only-dark)

!!! tip

    On UNIX-based systems, you can retrieve your SSH public key by running the following command in your terminal:
    ```
    cat ~/.ssh/id_rsa.pub
    ```

#### Anti Affinity setup

![title](./images/extra/configure_exoscale/anti_affinity_light.png#only-light)
![title](./images/extra/configure_exoscale/anti_affinity_dark.png#only-dark)

![title](./images/extra/configure_exoscale/anti_affinity_add_light.png#only-light)
![title](./images/extra/configure_exoscale/anti_affinity_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/anti_affinity_name_light.png#only-light)
![title](./images/extra/configure_exoscale/anti_affinity_name_dark.png#only-dark)

#### Instances setup

![title](./images/extra/configure_exoscale/instances_add_light.png#only-light)
![title](./images/extra/configure_exoscale/instances_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/instances_name_light.png#only-light)
![title](./images/extra/configure_exoscale/instances_name_dark.png#only-dark)

![title](./images/extra/configure_exoscale/instances_type_light.png#only-light)
![title](./images/extra/configure_exoscale/instances_type_dark.png#only-dark)

![title](./images/extra/configure_exoscale/instances_config_add_light.png#only-light)
![title](./images/extra/configure_exoscale/instances_config_add_dark.png#only-dark)

![title](./images/extra/configure_exoscale/instance_info_light.png#only-light)
![title](./images/extra/configure_exoscale/instance_info_dark.png#only-dark)

### How can I resume fine-tuning?

If your training process is interrupted or you want to continue from a previously saved checkpoint, you can enable checkpoint resumption in the configuration file.

Under the `SFTConfig` section, update the following parameter:

```yaml
# resume_from_checkpoint: false
resume_from_checkpoint: true
```

When this option is set to `true`, the `SFTTrainer` automatically detects the latest available checkpoint in your output directory and resumes training from that point, preserving model weights, optimizer states, and scheduler progress.

If you run the fine-tuning script as is, [`wandb`](https://wandb.ai/site/) will log a new run instead of continuing the previous one. To continue logging under the same initial run, you have two options:

To keep logging under the same initial run, set the environment variable before launching the script:

```bash
WANDB_RUN_ID="YOUR RUN ID" uv run accelerate launch ...
```

Alternatively, you can edit the [`.env`](.env) file and add the same entry:

```bash
WANDB_RUN_ID="YOUR RUN ID"
```

You can find your [`wandb`](https://wandb.ai/site/) run ID by opening the corresponding run page, navigating to Overview &#8594; Run path, and copying the identifier that appears after the `/`.