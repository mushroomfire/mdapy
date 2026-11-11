#!/usr/bin/env bash
# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# Asks public resolvers for the Ubuntu archive when the resolver of the runner
# cannot find it, as on 2026-11-11, so that apt-get still installs packages.
set -euo pipefail
if ! getent hosts security.ubuntu.com > /dev/null; then
  sudo mkdir -p /etc/systemd/resolved.conf.d
  printf '[Resolve]\nDNS=1.1.1.1 8.8.8.8\nDomains=~ubuntu.com\n' |
    sudo tee /etc/systemd/resolved.conf.d/ubuntu.conf > /dev/null
  sudo systemctl restart systemd-resolved
  getent hosts security.ubuntu.com
fi
