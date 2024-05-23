{ pkgs, ... }: {
  channel = "stable-23.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python311    
    pkgs.python311Packages.pip
    
    pkgs.python311Packages.matplotlib
  ];

  # Sets environment variables in the workspace
  env = {};
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.debugpy"
      "ms-python.python"
    ];

    previews = {
      enable = true;
    };
  };
}
